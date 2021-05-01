"""
Syntagmatic and Paradigmatic Word Embeddings 
from Co-occurence Matrix Decomposition
----------------------------------
renjith p ravindran 2021

Gives SVD factorized Low Rank Word Embeddings from
1) raw co-ccurrence frequency
2) Log co-occurrence frequency
3) Pointwise Mutual Information (PMI)
4) Positive PMI

SVD of data matrix X is given as,
X = U.S.VT
Where, U and VT are left and right sinular vectors
and S is a diagonal matrix of singular values

Low rank approximation of X is obtained by truncation of these
factor matrices at approprite ranks
X ~ X' = U'.S'.VT'
Where U' is  U truncated to rank (dimension) k, similarly S' and VT'

For standard LSA, U' is considered as the final embedding.
This corresponds to scaling the singular vectors as U'.(S'**p), where p=0.
Caron2001 showed that different exponential weights lead to embeddings
with a more softer/nuanced rank selection.
For example letting p>1 can give more weightage to leading singular vectors

The embeddings are saved in Word2Vec format
so that they may later be explored using Gensim

Most of the computation is done parallely.
However, due to the untyped nature of python
The memory requiremnts are pretty high.
More the number of cores you use, more memory you need.
"""

import numpy as np
from more_itertools import divide
from math import log2
import pickle
from pathlib import Path

from scipy.sparse import csr_matrix, coo_matrix
# saprsity heavy lifting||

import concurrent.futures
# concurrency heavy lifting!!

from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
# we use randomized_svd instead of TruncatedSVD.
# TruncatedSVD is what is generally used which calls randomized_svd within.
# TruncatedSVD gives VT, the right singular vectors, as the componenents of SVD,
# and not U , the left singular matrix. 
# X = U.S.VT
# We need to do U.S, but
# U.S = X.V, so we can get U.S by doing X.V, but numpy gives S as a list 
# and not a diagonal matrix, and it also provides broadcasting
# so U.S , matrix multiplication with a diagonal matrix,
# can be replace with multiplying each row with corresponding singular value,
# which is quickly done by broadcating.
# thus we may get the result of matrix multiplication without matrix meultiplication
# For this we need U and not VT
# Also, we need to do U.(S**p)
# randomized_svd gives all components, so we use that.


#--- global vars to offload class data for efficient access by concurrent futures.
# the data in these come from corresponding vars in class object
# but having concurrent futures access class varibales drastically impacts throughput
# concurrent futures passes  around data as pickle objects, doing that with 
# big class objects seems to add significant overhead.
# therefore before concurrent execution all the required variables
# are offloaded into global vars for the concurrent functions to read from,
# this gives seemingly full processor utilization.
corpus_ = []
coocs_raw_ = '' 
coocs_pmi_ = '' 
word_coocs_ = ''
total_coocs_ = ''
chunks_ = []
#--------

class SPVec:
    """
    Parameters
    ----------
    jobs : int, default=35
        no of parallel jobs
        
    corpus_filename : String 
        filename of a pre-processed corpus file.
        No processing such as minimum count of tokens 
        or any other filtering is done.
        
    model_filename : String
        filename of the pickled modelfile,
        made by this module.
        
        **either corpus_filename or model_filename must be given**
        **if corpusfile is given, consider the following parameters also**
        
    windowtype: String, default='par'
         'par'|'syn'
        
    windowsize : int, positive, default=4
        size of the co-occurrence window of windowtype
        
    Examples
    -------
    """
    def __init__(self,corpus_filename=None,model_filename=None,\
                 windowtype='par',windowsize=4,jobs=35,model_prefix='bnc'):
        """
        """
        if corpus_filename != None and windowsize > 0:
            self.jobs = jobs
            self.model_prefix = model_prefix
            self.corpus_filename = corpus_filename
            self.windowsize = windowsize
            self.windowtype =  windowtype
            self.word2id = {}
            # word to integer mapping
            self.id2word = {}
            # integer to word 
            self.vocabsize = 0
            self.corpus = []
            # in memory corpus with wordids
            # stored outside the class as only then will it work with parallel futures
            print("making vocab")
            self.make_vocab()
            print("..done")
            self.coocs_raw = csr_matrix((self.vocabsize,self.vocabsize),dtype = np.dtype('u4'))
            # sparse matrix for storing raw co-occurence frequencies
            self.coocs_pmi = csr_matrix((self.vocabsize,self.vocabsize),dtype = np.dtype('f4'))
            # for soring PMI weighted co-occurrences
            self.coocs_ppmi = csr_matrix((self.vocabsize,self.vocabsize),dtype = np.dtype('f4'))
            # for soring PMI weighted co-occurrences
            
            self.word_coocs = {}
            self.total_coocs = 0
            #
            self.tsvd_factors = {}
            #
            print("counting co-occurrences")
            self.count_coocs()
            print("..done")
    
            
        elif model_filename is not None:
            self.load_cooc_model(model_filename)
        else:
            raise ValueError
        
        
    def make_vocab(self):
        """
        assigns integer ids to each word in the corpus
        and stores corpus in memory as list of list of ids
        """
        wordid=0
        with open(self.corpus_filename) as file_:
            for line in file_:
                line = line.strip().split()
                # get only alphanumerics
                
                line_=[]
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
                
        self.vocabsize=len(self.word2id)

        
    
    def count_coocs(self):
        """
        parallel counting of co-occurences using parallel futures
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
                #csr matrix
                self.coocs_raw += coocs_chunk
                #adding csr matrices
        
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
        
        print("computing pmi")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            vocab_chunks = divide(self.jobs,range(self.vocabsize))
            
            futures = {executor.submit(pmi_worker1,chunk) for chunk in vocab_chunks}
            for future in concurrent.futures.as_completed(futures):
                word_coocs,total_coocs = future.result()
                self.word_coocs.update(word_coocs)
                self.total_coocs += total_coocs
        
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
        
        coocs_raw_ = ''
        word_coocs_ = ''
        total_coocs_ = ''
        chunks_ = ''
        
        print("..done")
        
    def make_ppmiMat(self):
        """
        Once we have the PMI matrix we can simply iterate over nonzero values
        and keep just the postive values and we get the PPMI matrix.
        We do that in parallel.
        """
        global chunks_
        
        ppmi_vals = []
        ppmi_rows = []
        ppmi_cols = []
        
        if self.coocs_pmi.data.nbytes == 0:
            #check if pmi is computed yet
            self.make_pmiMat()
        
        print("transforming to ppmi")
        
        pmi_coo =  self.coocs_pmi.tocoo()
        # from csr to coo format, this is fast, 500ms
            
        chunks_=  divide(self.jobs,zip(pmi_coo.data,pmi_coo.row,pmi_coo.col))
        # store chunks of coo outside for efficeint acces with concurent futures
        
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
        # and finally to csr
        chunks_ = ''
        #resetting 
        
        print("..done")
        
    def factorize(self,data_matrix,rank):
        """
        using sk-learn randomized svd
        """
        
        print("factorizing")
        u,s,vt = randomized_svd(data_matrix,rank) 
        print("..done")
        return u,s,vt
        
    def make_lr_embeddings(self,save_to=None,term_weight='pmi',dim=100,p=1):
        """
        make low rank embeddings using truncated SVD
        
        parameters
        ---------
        save_to: String, default=None
            filname of embeding file
            if None return numpy array
        term-weight: String, allowed=raw|log|pmi|ppmi, default=pmi 
        dim: int, default=100
            required dimentionaility of the embedding space
        p: int, default=1
            scaling power factor of the eigen vectors
            if X = U.S.VT, the final embedding will be
            U.pow(S,p)
            
        """
        
        if term_weight == "raw":
            if ('raw',dim) not in self.tsvd_factors:
                # check if we have factorized yet
                u,s,vt = self.factorize(self.coocs_raw.todense(),dim)
                self.tsvd_factors[('raw',dim)] = (u,s,vt)
                # save the factors
            u,s,vt = self.tsvd_factors[('raw',dim)]
            
        elif term_weight == "log":
            if ('log',dim) not in self.tsvd_factors:
                u,s,vt = self.factorize(self.coocs_raw.log1p().todense(),dim)
                self.tsvd_factors[('log',dim)] = (u,s,vt)
            u,s,vt = self.tsvd_factors[('log',dim)]
            
        elif term_weight == "pmi":
            if self.coocs_pmi.data.nbytes == 0:
                #check if pmi is not computed yet
                self.make_pmiMat()
                u,s,vt = self.factorize(self.coocs_pmi.todense(),dim)
                self.tsvd_factors['pmi'] = (u,s,vt)
            elif ('pmi',dim) not in self.tsvd_factors:
                #pmi is computed but not factorized yet
                u,s,vt = self.factorize(self.coocs_pmi.todense(),dim)
                self.tsvd_factors[('pmi',dim)] = (u,s,vt)
            else:
                #pmi is computed and also factorized
                u,s,vt = self.tsvd_factors[('pmi',dim)]
                
        elif term_weight == 'ppmi':
            if self.coocs_ppmi.data.nbytes == 0:
                #check if ppmi is not computed yet
                self.make_ppmiMat()
                u,s,vt = self.factorize(self.coocs_ppmi.todense(),dim)
                self.tsvd_factors['ppmi'] = (u,s,vt)
            elif ('ppmi',dim) not in self.tsvd_factors:
                #ppmi is computed but not factorized yet
                u,s,vt = self.factorize(self.coocs_ppmi.todense(),dim)
                self.tsvd_factors[('ppmi',dim)] = (u,s,vt)
            else:
                #ppmi is computed and also factorized
                u,s,vt = self.tsvd_factors[('ppmi',dim)]
            
        else:
            raise ValueError("un-recognized term-weight")
        
        print("making embeddings")
        s = s**p
        #scaling eighen values with scale factor
        # 
        if self.windowtype == 'par':
            emb = u*s
        elif self.windowtype == 'syn':
            emb_l = u*s
            emb_r = s*vt.transpose()
        else:
            raise ValueError
        #final embeddings
        print("..done")
        param_str = "{}_wt:{}_ws:{}_tw:{}_dim:{}_p:{}_".format(self.model_prefix,self.windowtype,
                                 self.windowsize,term_weight,dim,p) 
        if save_to != None:
            if not Path(save_to).is_dir():
                save_to = "./" 
            filename = str(Path(save_to,param_str))+".vec"
            if self.windowtype ==  'par':
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
        """
        word_count,dim = emb_l.shape
        
        print("saving embeddings to disk at",filename)
        
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
                
    
    def save_w2v_format_par(self,emb,filename="file.vec"):
        """
        """
        word_count,dim = emb.shape
            
        print("saving embeddings to disk at",filename)
        
        with open(filename,"w") as f_:
            print(word_count,dim,file=f_)
            for i in range(word_count):
                word = self.id2word[i]
                
                print(word,end=' ',file=f_)
                for j in range(dim):
                    print("{:.5f}".format(emb[i,j]),end=" ",file=f_)
                print(file=f_)
                
        print("..done")
        
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
    # which includes invalid postions such as position i itself
    # and positions that may not be available in a given sentence (<0 and >len(sentence))
    # we remove these invalid postions later
        
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
    """
    global coocs_raw_
    word_coocs = {}
    total_coocs = 0
    for i in vocab_chunk:
        word_coocs[i] = np.sum(coocs_raw_[i,:])
        # summing rows in csr format is fast
        total_coocs += word_coocs[i]
        # adding each row sum to get total sum
    return word_coocs,total_coocs
    
def pmi_worker2(chunk_id,vocabsize):
    """
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



        