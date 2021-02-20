"""
Evaluation of SPVec embeddings on SP-10K dataset
-----------------
renjith p ravindran 2021
"""
import re
from glob import glob
from pathlib import Path

from collections import namedtuple
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from gensim.models import KeyedVectors

class SP_Eval:
    """
    usage
    ------
    Eval = SP_Eval()
    Eval.load_model_files(...)
    Eval.evaluate()
    Eval.plot_results(...)
    """
    
    def __init__(self,sp10k_path='SP-10K/data',exclude=['wino']):
        
        self.param_text={
            'ws':'window-size',
            'dim':'dimension',
            'tw':'term-weight',
            'p':'exponential-weight'
        }
        
        self.modelfiles = {}
        self.modelresults = {}
        
        self.sp_dataset = self.load_sp10k(sp10k_path,exclude)
        
    
    def load_sp10k(self,path,exclude):
        """
        parameters
        -----
        path : str, default='SP-10K/data'
            the path to the directory that contains SP-10K dataset files
        
        exclude : list of str, default=['wino']
            specify the sp10k categories to exclude
            use this to exclude winograd files
        """
        sp10k_data = {}
        file_names = glob(path+'/*annotation*')
        
        for file_name in file_names:
            cat = file_name.split("/")[2].replace("_annotation.txt",'')
            # sp-10k categories: 'dobj','amod','nsubj','nsubj_amod','dobj_amod',
            # 'wino_dobj_amod', 'wino_nsubj_amod'
            
            if any([exclude_str in cat for exclude_str in exclude]):
                # exclude any category
                # basically to exclude the winograd files
                continue
                
            items = []
            with open(file_name) as file_:
                for line in file_:
                    word1,word2,score = line.split()
                    items.append((word1,word2,float(score)))
                    sp10k_data[cat] = items
        return sp10k_data 
    
    def load_model_files(self,path,model_class,index_params,having_params=None):
        """
        add multiple model files of a model class.
        eg. to add all word2vec models
        this can be called multiple times to add models of different classes
        
        params
        -----
        path : str
            the path to a directory with many model files
        
        model_class : str
            some name to group a set of models, eg. word2vec, fasttext, etc
        
        index_params : list
            important parameters that can index the model, eg. {'win':100}
            this allows to study the effect of these params on the SP-10K 
  
        having_params: dict, default=None
            keep only model files having specified parameters, eg. {'tw':'ppmi'}
        """
        if 'wt' not in index_params:
            # this is always reuired to differentiate syntagmatic and 
            # paradimatic embeddings
            index_params.append('wt')
            
        if having_params != None:
            having_params = [key+":"+value for key,value in having_params.items()]
            # files with these in their filenames should be included
        else:
            having_params = []
            
        
        if model_class not in self.modelfiles.keys():
            # check if model class exists
            self.modelfiles[model_class] = []
            
        if Path(path).is_dir():
            filenames = glob(str(Path(path,"*.vec")))
            
            for filename in filenames:
                if all([param_str in filename for param_str in having_params]):
                    # keep only the selected files
                    params = {}
                    for paramname in index_params:
                        
                        regex = re.compile(paramname+":(.+?)_")
                        
                        try:
                            paramvalue = regex.search(filename).group(1)
                        except AttributeError:
                            paramvalue = 'par'
                            
                        params[paramname] = paramvalue
                        
                    self.modelfiles[model_class].append((params,filename))
        else:
            raise ValueError('only directories, not regular files')
        
        
    def evaluate(self):
        """
        compute correlation with every category in sp10k with all models
        """
        Result = namedtuple('Result',['indexes','correlation','significance'])
        for model_class in self.modelfiles:
            for indexes,filename in self.modelfiles[model_class]:
                model_type = indexes['wt']
                model = SP_Model(filename,model_type)
                print(filename)
                
                oovs=set()
                results = []
                for cat in self.sp_dataset.keys():
                    model_scores=[]
                    # scores from model
                    gt_scores=[]
                    # ground thruth scores
                    for item in self.sp_dataset[cat]:
                        word1,word2,gt_score=item
                        word1=word1.lower()
                        word2=word2.lower()
                        if not model.in_vocab(word1):
                            oovs.add(word1)
                            continue
                        if not model.in_vocab(word2):
                            oovs.add(word2)
                            continue
                        model_scores.append(model.score(word1,word2,cat))
                        gt_scores.append(gt_score)
                    correlation,significance=spearmanr(model_scores,gt_scores)
                    print("correlation:",correlation,
                          "significance",significance,
                          "oovs:",len(oovs))
                    
                    result = Result(indexes,correlation,significance)
                    results.append((cat,result))
                    
                if model_class not in self.modelresults:
                    self.modelresults[model_class] = {}
                    
                for cat,result in results:
                    if cat not in self.modelresults[model_class]:
                        self.modelresults[model_class][cat] = []
                    self.modelresults[model_class][cat].append(result)

    
    def plot_results(self, model_class, x_axis_index='ws', averaged=False, plot_title=None, ignore_cat=None):
        """
        plot correlations  for each model in the given model class.
        Every line in the plot is a SP-10K category.
        The x-axis values are the paramvalues of the index parameter of different models in the class
        The y-axis is the spearman's correlation
        
        Thus if yu have 5 model under the model class, each with index parameter as window-size
        then x-axis refers to a specific model with given window-size,
        and the y-axis the corresponding correlation value.
        
        parameters
        ------
        model_class : str
            the same name given during `load_model_files`
        
        x_axis_index : str, default='ws'
            the index parameter for the x-axis
        
        averaged : bool, default=False
            Plot average correlation over all SP-10K categories
            
        plot_title : str, default=None
            if None the title is "SP-10K EValuation on "+model_class
        
        ignore_cat : list, default=None
            speficify categories to ignore, eg. ['dobj_amod','nsubj_amod']
            
        """
        model_classes = [] 
        
        if model_class == 'all':
            model_classes = self.modelfiles.keys()
        else:
            model_classes = [model_class]
       
        if ignore_cat is None:
            ignore_cat = []
            
        ys_all = []
        for model_class_ in model_classes:
            
            xs = []
            for cat,results in self.modelresults[model_class_].items():
                if cat in ignore_cat:
                    continue
                xs = []
                ys = []
                for result in results:
                    ys.append(result.correlation)
                    xs.append(result.indexes[x_axis_index])
                xs_ys = list(zip(xs,ys))
                xs_ys = sorted(xs_ys,key=lambda xy: xy[0])
                xs,ys = zip(*xs_ys)
                if averaged is False:
                    plt.plot(xs,ys,label=model_class_+":"+cat,marker='o')
                else:
                    ys_all.append(ys)
            
            if averaged:
                ys_cat_avg = [sum([row[i] for row in ys_all]) for i in range(len(ys_all[0]))]
                ys_cat_avg = [sum_/len(ys_all) for sum_ in ys_cat_avg]
                plt.plot(xs,ys_cat_avg,label=model_class_,marker='o')
                
                
        plt.xlabel(self.param_text[x_axis_index])
        plt.ylabel("spearmans-correlation")
        
        if model_class == 'all':
            model_str = ','.join(model_classes)
        else:
            model_str = model_classes[0]
        
        if plot_title is None:
            plt.title("SP-10K Spearmans Correlation on "\
                          +model_str+"\nwith varying "+self.param_text[x_axis_index])
        else:
            plt.title(plot_title)
            
        plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
        
    
class SP_Model:
    """
    select different score functions based on embedding type
    syntagmatic or paradimatic
    
    Synatagmatic embeddings have 2 embeddings per word.
    One of left context and the other of right context.
    To measure the syntagmatic association between two words in sequence; w1 w2
    We compute the cosine similarity between the right embedding of w1 
    and left embedding of w2.
    I would like to use gensim for all such computation, since it is really fast.
    However, gensim does not understand the idea multiple embeddings per word.
    To work with gensim and word2vec format of embedding files
    we treat both left and right embedding of a word as seperate word.
    By suffixing '_l' and '_r' to the correspoding word and embedding.
    Now gensim can seen all the embedings in the same space, and do its magic.
    
    """
    
    def __init__(self,emb_file,emb_type='syn'):
        """
        """
        self.model = KeyedVectors.load_word2vec_format(emb_file,binary=False)
        self.model_type = emb_type
    
    def in_vocab(self,word):
        if self.model_type == 'syn':
            word = word+'_l'
            
        if word in self.model.vocab.keys():
            return True
        else:
            return False
    
    def score(self,word1,word2,cat=None):
        
        if self.model_type == "syn":
            return self.syn_score(word1,word2,cat)
        elif self.model_type == "par":
            return self.par_score(word1,word2)
        else:
            raise ValueError
    def par_score(self,word1,word2):
        """
        """
        score = 1 - self.model.distance(word1,word2)
        
        return score
        
        
    def syn_score(self,word1,word2,cat):
        """
        word1 is the base word
        word2 is either an amod, nsubj or dobj
        """
        if cat == 'amod':
            # word2 occurs to the left of word1
            word1 = word1+"_l"
            word2 = word2+"_r"
        elif cat == 'nsubj':
            # word2 occurs to the left of word1
            word1 = word1+"_l"
            word2 = word2+"_r"
        elif cat == 'dobj':
            # word 2 occurs to the right of word1
            word1 = word1+"_r"
            word2 = word2+"_l"
        elif cat == 'nsubj_amod':
            # word2 occurs to the left of word1
            word1 = word1+"_l"
            word2 = word2+"_r"
        elif cat == 'dobj_amod':
            # word 2 occurs to the right of word1
            word1 = word1+"_r"
            word2 = word2+"_l"
        else:
            raise ValueError
            
        score=1 - self.model.distance(word1,word2)
        return score    
    