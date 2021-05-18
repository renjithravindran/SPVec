"""
SPVec: Syntagmatic and Paradigmatic Word Embeddings 
from Co-occurrence Matrix Decomposition
----------------------------------
renjith p ravindran 2021

Syntagmatic representations are directional(left/right) word
embeddings, each word thus has a left embedding and a right
embedding.  Left embedding (w.l) of a word is the
representation of a word with respect to words that appear
to its left and similarly right embedding (w.r) represents a
words right context.

To get the syntagmatic association of w1 to the left of w2,
perform cosine similarity between w1.r and w2.l

Syntagmatic representations are useful for learning
selectional preference (SP), and since SPVec might mostly be
used for learning SP, SPVec can also stand for Selectional
Preference Vectors

Paradigmatic representations are similar to embeddings such
as word2vec and glove, where the context is symmetric and
thus  there is no distinction between left and right
context, yielding just one embedding per word

SPVec uses SVD to obtain word embeddings as low rank 
representations of the word co-occurrence matrix.
The co-occurrence matrix can be of:
1) raw co-occurrence frequency 
2) Log co-occurrence frequency 
3) Pointwise Mutual Information (PMI) 
4) Positive PMI

SVD of data matrix X is given as, X = U.S.VT Where, U and VT
are left and right singular vectors and S is a diagonal
matrix of singular values

Low rank approximation of X is obtained by truncation of
these factor matrices at appropriate ranks (factors), 
X ~ X' = U'.S'.VT' Where X' is the low-rank approximation of X and
U' and VT' are truncated singular vector matrices and S' the
truncated singular values metrics.

The embeddings are saved in Word2Vec format so that they may
later be explored using Gensim

Most of the computation is done parallel.  However, due to
the untyped nature of python the memory requirements are
pretty high.  More the number of cores you use, more memory
you need.

NOTE: 
1)  Currently SPVec expects a pre-processed corpus.
    Common corpus processing such as lower-casing,
    numerals-removal, min-counting etc should be done in prior.
2)  The corpus should be small enough to be loaded to RAM

"""

import numpy as np
from more_itertools import divide
from math import log2
import pickle
from pathlib import Path
from timeit import default_timer
from datetime import timedelta

from scipy.sparse import csr_matrix, coo_matrix
# sparsity heavy lifting!!

import concurrent.futures
# concurrency heavy lifting!!

from sklearn.utils.extmath import randomized_svd
# we use randomized_svd instead of TruncatedSVD.
# TruncatedSVD is what is generally used which calls randomized_svd within.
# TruncatedSVD gives VT, the right singular vectors, as the components of SVD,
# and not U , the left singular matrix (X = U.S.VT)
# We need both left and right singular vectors therefore we
# use randomized_svd which gives all the components.
# however, randomised_svd does not work on sparse matrices
# that we use to get the co-occurrence counts.

from sparsesvd import sparsesvd
# the co-occurrence matrix may be bigger than RAM capacity
# for such cases we use sparsesvd, but sparsesvd is slower than randomized_svd

#--- global vars for efficient access by concurrent futures ---
# the data in these come from corresponding vars in class object but having
# concurrent futures access class variables drastically impacts throughput.
# concurrent futures passes  around data as pickle objects, doing that with big
# class objects seems to add significant overhead.  Therefore before concurrent
# execution all the required variables are offloaded into global vars for the
# concurrent functions to read from, this gives seemingly full processor
# utilization.

corpus_ = []
coocs_raw_ = '' 
coocs_pmi_ = '' 
word_coocs_ = ''
total_coocs_ = ''
chunks_ = []
#---


class SPVec:
    """
    Example Usage
    -------
    >>> spvec = SPVec(corpus_file='pre-processed-bnc.txt', windowtype='syn',
            windowsize=3, jobs=5, model_prefix='spvec_bnc')
    >>> spvec.make_lr_embeddings()
    """
    
    def __init__(self,corpus_filename=None,model_filename=None,\
                 windowtype='par',windowsize=3,jobs=10,model_prefix='spvec'):
        """
        Parameters
        ----------
        jobs : int, default=10
            no of parallel jobs
            
        corpus_filename : String 
            filename of a pre-processed corpus file.
            No processing such as minimum count of tokens 
            or any other filtering is done.
            
        model_filename : String
            filename of the pickled modelfile,
            made by this module.
            
            **either corpus_filename or model_filename must be given**
            **if corpus_filename is given, consider the following parameters also**
            
        windowtype: String, default='par'
             'syn': syntagmatic vectors
             'par': paradigmatic vectors
            
        windowsize : int, positive, default=3
            size of the co-occurrence window

        modelprefix: String, default='spvec'
            by default when a word embeddings are saved to disk
            all the parameter-values are formatted to make the filename
            this allows to extract model parameters from the filename later.
            modelprfix is the string that prefixes the filename,
            this may be used to mark the corpus used for learning the embeddings
            eg. if you use the BNC corpus, set the modelprefix as 'spvec_bnc'
        """

        if corpus_filename is not None and model_filename is None:
            assert windowsize > 0
            self.jobs = jobs
            self.model_prefix = model_prefix
            self.corpus_filename = corpus_filename
            self.windowsize = windowsize
            self.windowtype =  windowtype

            self.word2id = {}
            # word to integer mapping
            self.id2word = {}
            # integer to word mapping
            self.vocabsize = 0
            self.corpus = []
            # in memory corpus with wordids
            # stored outside the class as only then will it work with parallel futures 

            print("making vocab")
            starttime =  default_timer()
            self.make_vocab()
            delta =  default_timer() - starttime
            print("..done ({})".format(timedelta(seconds=delta)))

            shape = (self.vocabsize,self.vocabsize)
            self.coocs_raw = csr_matrix(shape,dtype = np.dtype('u4'))
            # sparse matrix for storing raw co-occurrence frequencies
            self.coocs_pmi = csr_matrix(shape,dtype = np.dtype('f4'))
            # for storing PMI weighted co-occurrences
            self.coocs_ppmi = csr_matrix(shape,dtype = np.dtype('f4'))
            # for storing PPMI weighted co-occurrences
            
            #--- some memoization for PMI computations
            self.word_coocs = {}
            self.total_coocs = 0
            #---

            self.tsvd_factors = {}
            # keeping factor matrices from truncated SVD

            print("counting co-occurrences")
            starttime = default_timer()
            self.count_coocs()
            delta = default_timer() - starttime
            print("..done ({})".format(timedelta(seconds=delta)))
    
        elif model_filename is not None:
            self.load_cooc_model(model_filename)
        else:
            raise ValueError("provide either corpus-filename or model-filename")
        
        
    def make_vocab(self):
        """
        assigns integer ids to each word in the corpus
        and stores corpus in memory as list of list of ids
        """

        wordid = 0
        with open(self.corpus_filename) as file_:
            for line in file_:
                line = line.strip().split()
                # simple tokenize
                
                line_ = []
                # line with wordids, for in-memory corpus
                
                if len(line) == 1:
                    # no co-occurrence here!
                    continue
                    
                for word in line:
                    if word not in self.word2id:
                        self.word2id[word] = wordid
                        self.id2word[wordid] = word
                        wordid += 1
                    line_.append(self.word2id[word])
                    
                self.corpus.append(line_)
                # the corpus is stored because file reading is slow 
                # and co-occurrence counting requires lots of reads
                
        self.vocabsize = len(self.word2id)

        
    def count_coocs(self):
        """
        parallel counting of co-occurrences using parallel futures
        each parallel worker is given a range of line numbers 
        of the corpus to work on
        """
        
        global coocs_raw_
        global chunks_
        global corpus_
        
        corpus_ = self.corpus
        # offloading
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            chunks_=[list(lines) for lines in divide(self.jobs,range(len(self.corpus)))]
            ws = self.windowsize
            vs = self.vocabsize
            wt = self.windowtype
            
            futures={executor.submit(coocs_worker,chunk_id,ws,wt,vs)\
                                         for chunk_id in range(len(chunks_))}
            for future in concurrent.futures.as_completed(futures):
                coocs_chunk=future.result()
                # csr matrix
                self.coocs_raw += coocs_chunk
                # adding csr matrices  to get total co-occurrences
                # currently this is done sequentially, parallel reduce would be great!
        
        corpus_ = ''
        # resetting
   

    def make_pmiMat(self):
        """
        """

        global word_coocs_
        global total_coocs_
        global coocs_raw_
        global coocs_pmi_
        global chunks_
        
        coocs_raw_ = self.coocs_raw
        # offloading

        print("computing pmi")
        starttime = default_timer()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            vocab_chunks = divide(self.jobs,range(self.vocabsize))
            
            futures = {executor.submit(pmi_worker1,chunk) for chunk in vocab_chunks}
            for future in concurrent.futures.as_completed(futures):
                word_coocs,total_coocs = future.result()
                self.word_coocs.update(word_coocs)
                self.total_coocs += total_coocs
        
        #--- offloading
        word_coocs_ = self.word_coocs
        total_coocs_ = self.total_coocs
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            rows,cols = coocs_raw_.nonzero()
            nonzeros =  zip(rows,cols)
            chunks_=  [list(nz) for nz in divide(self.jobs,nonzeros)]
            futures = {executor.submit(pmi_worker2,chunk_id,self.vocabsize)\
                       for chunk_id in range(len(chunks_))}
            for future in concurrent.futures.as_completed(futures):
                pmi_csr = future.result()
                self.coocs_pmi += pmi_csr 
       
        #--- resetting
        coocs_raw_ = ''
        word_coocs_ = ''
        total_coocs_ = ''
        chunks_ = ''
        
        delta = default_timer() - starttime
        print("..done ({})".format(timedelta(seconds=delta)))
   

    def make_ppmiMat(self):
        """
        Once we have the PMI matrix we can simply iterate over nonzero values
        and keep just the positive values to get the PPMI matrix.
        We do that in parallel.
        """
        
        global chunks_
        
        ppmi_vals = []
        ppmi_rows = []
        ppmi_cols = []
        
        if self.coocs_pmi.data.nbytes == 0:
            # check if pmi is computed yet
            self.make_pmiMat()
        
        print("transforming to ppmi")
        starttime = default_timer()

        pmi_coo =  self.coocs_pmi.tocoo()
        # from csr to coo format, this is fast, 500ms (bnc min-count==100)
            
        chunks_=  divide(self.jobs,zip(pmi_coo.data,pmi_coo.row,pmi_coo.col))
        # store chunks of coo outside class for efficient access with concurrent futures
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(ppmi_worker,chunk_id)\
                       for chunk_id in range(len(chunks_))}
            for future in concurrent.futures.as_completed(futures):
                vals,rows,cols = future.result()
                
                ppmi_vals.extend(vals)
                ppmi_rows.extend(rows)
                ppmi_cols.extend(cols)
                
        ppmi_coo = coo_matrix((ppmi_vals,(ppmi_rows,ppmi_cols)),\
                         shape=(self.vocabsize,self.vocabsize),dtype=np.dtype('f4'))
        # make ppmi in coo format
        
        self.coocs_ppmi = ppmi_coo.tocsr()
        # and finally back to csr

        chunks_ = ''
        #resetting 
        
        delta = default_timer() - starttime
        print("..done ({})".format(timedelta(seconds=delta)))
    

    def factorize(self,data_matrix,num_factors,sparse=False):
        """
        perform truncated svd using either sklearn's randomized_svd
        or Radim's sparsesvd
        """

        starttime = default_timer()

        if sparse:
            print("factorizing sparse")
            u,s,vt = sparsesvd(data_matrix.tocsc(),num_factors)
            #u is actually ut
            u = u.transpose()
        else:
            print("factorizing dense")
            u,s,vt = randomized_svd(data_matrix.todense(),num_factors) 
        
        delta = default_timer() - starttime
        print("..done ({})".format(timedelta(seconds=delta)))

        return u,s,vt
    

    def make_lr_embeddings(self,save_to="./",term_weight='log',dim=300,p=0.5,sparse=False):
        """
        make low rank embeddings using truncated SVD
        
        parameters
        ---------
        save_to: String, default="./"
            path to directory, or path to a file.
            if directory, the filename is computed from the model parameter-values.
            if None return numpy array


        term-weight: String, allowed=raw|log|pmi|ppmi, default=log

        dim: int, default=300
            required dimensionality of the embedding space

        p: int, default=0.5
            scaling power factor of the singular vectors
            
        sparse: bool, default=False
            if true uses sparsesvd for factorisation
            else uses randomized_svd from sklearn
        """

        if term_weight == "raw":
            if ('raw',dim) not in self.tsvd_factors:
                # check if we have factorized yet
                u,s,vt = self.factorize(self.coocs_raw,dim,sparse=sparse)
                self.tsvd_factors[('raw',dim)] = (u,s,vt)
                # save the factors
            u,s,vt = self.tsvd_factors[('raw',dim)]
            
        elif term_weight == "log":
            if ('log',dim) not in self.tsvd_factors:
                u,s,vt = self.factorize(self.coocs_raw.log1p(),dim,sparse=sparse)
                self.tsvd_factors[('log',dim)] = (u,s,vt)
            u,s,vt = self.tsvd_factors[('log',dim)]
            
        elif term_weight == "pmi":
            if self.coocs_pmi.data.nbytes == 0:
                #check if pmi is not computed yet
                self.make_pmiMat()
                u,s,vt = self.factorize(self.coocs_pmi,dim,sparse = sparse)
                self.tsvd_factors['pmi'] = (u,s,vt)
            elif ('pmi',dim) not in self.tsvd_factors:
                #pmi is computed but not factorized yet
                u,s,vt = self.factorize(self.coocs_pmi,dim,sparse = sparse)
                self.tsvd_factors[('pmi',dim)] = (u,s,vt)
            else:
                #pmi is computed and also factorized
                u,s,vt = self.tsvd_factors[('pmi',dim)]
                
        elif term_weight == 'ppmi':
            if self.coocs_ppmi.data.nbytes == 0:
                #check if ppmi is not computed yet
                self.make_ppmiMat()
                u,s,vt = self.factorize(self.coocs_ppmi,dim,sparse = sparse)
                self.tsvd_factors['ppmi'] = (u,s,vt)
            elif ('ppmi',dim) not in self.tsvd_factors:
                #ppmi is computed but not factorized yet
                u,s,vt = self.factorize(self.coocs_ppmi,dim,sparse = sparse)
                self.tsvd_factors[('ppmi',dim)] = (u,s,vt)
            else:
                #ppmi is computed and also factorized
                u,s,vt = self.tsvd_factors[('ppmi',dim)]
            
        else:
            raise ValueError("un-recognized term-weight")
        
        print("making embeddings")
        starttime = default_timer()

        s = s**p
        # scaling eigen values with scale factor
        
        if self.windowtype == 'par':
            emb = u*s
        elif self.windowtype == 'syn':
            emb_l = u*s
            emb_r = s*vt.transpose()
        else:
            raise ValueError

        delta = default_timer() - starttime
        print("..done ({})".format(timedelta(seconds=delta)))

        param_str = "{}_wt:{}_ws:{}_tw:{}_dim:{}_p:{}_".format(self.model_prefix,self.windowtype,
                                 self.windowsize,term_weight,dim,p) 
        if save_to != None:
            if Path(save_to).is_dir():
                filename = str(Path(save_to,param_str))+".vec"
            else:
                filename = save_to

            if self.windowtype == 'par':
                self.save_w2v_format_par(emb,filename)
            else:
                self.save_w2v_format_syn(emb_l=emb_l,emb_r=emb_r,filename=filename)
        else:
            if self.windowtype == 'par':
                return emb
            else:
                return emb_l,emb_r
       

    def save_w2v_format_syn(self,emb_l,emb_r,filename="file.vec"):
        """
        save syntagmatic embeddings in word2vec format
        """

        word_count,dim = emb_l.shape
        
        print("saving embeddings to disk at",filename)
        starttime = default_timer()

        with open(filename,"w") as f_:
            print(word_count*2,dim,file=f_)
            # since we have both left and right embedding for each word
            for i in range(word_count):
                word = self.id2word[i]
                
                print(word+"_l",end=' ',file=f_)
                for j in range(dim):
                    print("{:.5f}".format(emb_l[i,j]),end=" ",file=f_)
                print(file=f_)
                
                print(word+"_r",end=' ',file=f_)
                for j in range(dim):
                    print("{:.5f}".format(emb_r[i,j]),end=" ",file=f_)
                print(file=f_)
                
        delta = default_timer() - starttime
        print("..done ({})".format(timedelta(seconds=delta)))
   

    def save_w2v_format_par(self,emb,filename="file.vec"):
        """
        save paradigmatic embeddings in word2vec format
        """

        word_count,dim = emb.shape
            
        print("saving embeddings to disk at",filename)
        starttime = default_timer()

        with open(filename,"w") as f_:
            print(word_count,dim,file=f_)
            for i in range(word_count):
                word = self.id2word[i]
                
                print(word,end=' ',file=f_)
                for j in range(dim):
                    print("{:.5f}".format(emb[i,j]),end=" ",file=f_)
                print(file=f_)
                
        delta = default_timer() - starttime
        print("..done ({})".format(timedelta(seconds=delta)))
        

    def save_cooc_model(self,model_filename='model'):
        """
        """

        with open(model_filename,"wb") as f_:
        
            pickle.dump(self.corpus_filename,f_)
            pickle.dump(self.windowsize,f_)
            pickle.dump(self.word2id,f_)
            pickle.dump(self.id2word,f_)
            pickle.dump(self.coocs,f_)
   

    def load_cooc_model(self,model_filename='model'):
        """
        """

        with open(model_filename,"rb") as f_:
            self.corpus_filename =  pickle.load(f_)
            self.windowsize =  pickle.load(f_)
            self.word2id =  pickle.load(f_)
            self.id2word = pickle.load(f_)
            self.coocs = pickle.load(f_)
            
            self.vocabsize = len(self.word2id)
            self.coocMat = np.zeros((self.vocabsize,self.vocabsize),dtype=np.dtype('u4'))
            print("making PMI")
            self.make_pmiMat()
            

def coocs_worker(chunk_id,windowsize,windowtype,vocabsize):
    """
    parallel worker using coo matrix
    """

    coocs_i=[]
    coocs_j=[]

    global chunks_
    
    if windowtype == 'par':
        window = lambda i : range(i - windowsize, i + windowsize +1)
    elif windowtype == 'syn':
        window = lambda i : range(i - windowsize, i + 1)
    else:
        raise ValueError
    # here window gives the co-occurrence positions j for a given position i
    # which includes invalid positions such as position i itself
    # and positions that may not be available in a given sentence (<0 and >len(sentence))
    # we remove these invalid positions later

    for line_no in chunks_[chunk_id]:
        line = corpus_[line_no]
        # reading from global variable
        for i in range(len(line)):
            for j in window(i):
                if j>=0 and j<= len(line) -1 and i!=j:
                    # only valid co-occurrence positions
                    coocs_i.append(line[i])
                    coocs_j.append(line[j])

    coocs_i=np.array(coocs_i)
    coocs_j=np.array(coocs_j)
    coocs_v=np.ones((coocs_i.shape[0],),dtype=np.dtype('u1'))
    coo_mat=coo_matrix((coocs_v,(coocs_i,coocs_j)),shape=(vocabsize,vocabsize),dtype=np.dtype('u4'))

    return coo_mat.tocsr()


def pmi_worker1(vocab_chunk):
    """
    This parallel worker does some memoization before PMI computation
    It does, total co-occurrence of a word ie. summing over all columns of a
    row, and total co-occurrence ie. sum of all word co-occurrences
    """

    global coocs_raw_
    word_coocs = {}
    total_coocs = 0
    for i in vocab_chunk:
        word_coocs[i] = np.sum(coocs_raw_[i,:])
        # summing rows in csr format is fast
        # giving total co-occurrence for each word
        total_coocs += word_coocs[i]
        # adding each row sum to get total sum
    return word_coocs,total_coocs


def pmi_worker2(chunk_id,vocabsize):
    """
    parallel worker for PMI computation
    """
    
    global total_coocs_
    global coocs_raw_
    global word_coocs_
    global chunks_
    
    pmi_vals = []
    rows = []
    cols = []
    
    for i,j in chunks_[chunk_id]:
        ij_cooc = coocs_raw_[i,j]

        i_cooc=word_coocs_[i]
        j_cooc=word_coocs_[j]
        pmi = log2(ij_cooc + 1) -log2(i_cooc + 1) -log2(j_cooc + 1) + log2(total_coocs_ + 1)
        
        
        rows.append(i)
        cols.append(j)
        pmi_vals.append(pmi)
        
    pmi_vals = np.array(pmi_vals)
    rows =  np.array(rows)
    cols = np.array(cols)
    
    pmi_coo = coo_matrix((pmi_vals,(rows,cols)),\
                         shape=(vocabsize,vocabsize),dtype=np.dtype('f4'))
    return pmi_coo.tocsr()


def ppmi_worker(chunk_id):
    """
    parallel worker for PPMI computation
    """
    
    global chunks_
    
    vals = []
    rows = []
    cols = []
    
    for val,row,col in chunks_[chunk_id]:
        if val > 0:
            # if positive we keep those
            vals.append(val)
            rows.append(row)
            cols.append(col)
    
    return vals,rows,cols


if __name__ == '__main__':


    spvec = SPVec(corpus_filename='bnc_mincount1000.txt', windowtype='syn',
            windowsize=3, jobs=1, model_prefix='spvec_bnc')
    spvec.make_lr_embeddings()
