"""
Evaluation of SPVEC embeddings
-----------------
renjith p ravindran 2021

This allows you to do the following

1) Querying embeddings for left/right associations

    Load multiple models and explore the syntagmtic associations
    from any of those models

    How-to: see docstring of class SP_Query

2) Evaluation on SP10K dataset

    Allows evaluations of multiple models(embeddings) at once.
    Each model may have different parameters, thus performance
    variations on different parameter values can be studied.
    Each model-file assumes parameter-values encoded in the filenames.

    Parameters are
    mt:(model-type)[syn/par]
    ws:(window-size)[1/2/3...]
    tw:(term-weight)[log/pmi/ppmi/raw]
    dim:(dimension)[...]
    p:(exponential-weight)[...]

    parameter-name and value is placed with ':' in between
    and every param-name-value is separated by '_'
    eg. `spvec_mt:syn_ws:3_tw:log_dim:300_p:0.5_.vec'

    Also allows evaluation of paradigmatic embeddings such as word2vec and glove.
    Just format the filenames appropriately to have the required parameter-values.
    All model-files should have .vec extension.
    Model-files without 'mt:syn' in the filename is assumed paradigmatic

    How-to: see docstring of class SP_Eval

TO-DO
-----
1) make this runnable as a script
2) run evaluations in parallel
"""
import re
from glob import glob
from pathlib import Path

from collections import namedtuple
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from more_itertools import divide

from gensim.models import KeyedVectors


class SP_Query:
    """
    Basic-usage
    -----------
    >>> from spvec_eval import SP_Query
    >>> spq = SP_Query(<path-to-embedding-file>)
    >>> spq.get_associations('food')
    """
    def __init__(self, model_path, having_params=None, max_models=4, paramnames=None):
        """
        Parameters
        ----------
        model_path : str
            path to model file or directory with many models

        having_params : list of str
            expects model-files to have parameter-values encoded in filename
            parameter-values separated by ':'
            eg: [ws:2,tw:log]

        max_models : int, default=4
            all models are loaded to memory at once
            so here we limit the number of models

        paramnames : list of str,default=['tw','mt','ws','dim','p','min']
            parameter-names encoded in model filename
        """

        self.models = {}
        self.models_selected = []
        # selected model filenames

        if having_params is None:
            having_params = []

        if paramnames is None:
            self.paramnames = ['tw', 'mt', 'ws', 'dim', 'p', 'min']
        else:
            self.paramnames = paramnames


        model_filenames = []
        models_selected = []


        if Path(model_path).is_dir():
            model_filenames = glob(str(Path(model_path, '*.vec')))
        else:
            model_filenames = [model_path]

        assert model_filenames, "no model-files found"

        for model_filename in model_filenames:
            if having_params:
                if all([param_value in model_filename for param_value in having_params]):
                    # select the model-files
                    self.models_selected.append(model_filename)
            else:
                self.models_selected.append(model_filename)

        assert self.models_selected, "no model-files satisfy selection"

        assert len(models_selected) <= max_models, "more than {} models selected".format(max_models)

        for model_filename in self.models_selected:
            params_values = self.get_model_params(model_filename)
            self.models[params_values] = KeyedVectors.load_word2vec_format(
                model_filename, binary=False)

    def get_model_params(self, filename):
        """
        """

        params_values = [part.split(":") for part in filename.split("_")]
        params_values = [part for part in params_values if len(part) == 2]
        params_values = [param+":"+value for param, value in params_values
                         if param in self.paramnames]

        return tuple(params_values)


    def get_associations(self, word, model_key=None, topn=50, print_=True):
        """
        Parameters
        --------
        word : str

        model_key : str
            the paramtername-value that can index the model
            eg. 'ws:2'
            model_key should index exactly one model

        topn : int, default=50
            how many associated words to be shown

        print_ : bool, default=True
            if False returns association dict
        """

        words_l = {}
        words_r = {}
        model = None
        associations = {}

        if model_key is None and len(self.models) > 1:
            raise ValueError("more than one model available, provide model_key ('name:value')")
        elif model_key is None and len(self.models) == 1:
            model = list(self.models.values())[0]

        elif model_key is not None:
            for model_params, model_ in self.models.items():
                for param_value in model_params:
                    if model_key == param_value:
                        if model is None:
                            model = model_
                        else:
                            raise ValueError("more than one model with", model_key)
        else:
            raise ValueError("This should not happen")

        assert model, "no model selected"

        for word_, sim in model.similar_by_word(word+'_l', topn=2000):
            if '_r' in word_:
                words_l[word_.replace('_r', '')] = sim
        for word_, sim in model.similar_by_word(word+'_r', topn=2000):
            if '_l' in word_:
                words_r[word_.replace('_l', '')] = sim

        words_l_ = set(words_l.keys())
        words_r_ = set(words_r.keys())

        wordsL = list(words_l_ - words_r_)
        wordsR = list(words_r_ - words_l_)
        # removing common words

        wordsL = sorted(wordsL, key=lambda x: words_l[x], reverse=True)
        wordsR = sorted(wordsR, key=lambda x: words_r[x], reverse=True)
        print("--------------")
        if print_:
            print("left:", ", ".join(wordsL[:topn]))
            print("")
            print("right:", ", ".join(wordsR[:topn]))
        else:
            associations['left'] = wordsL[:topn]
            associations['right'] = wordsR[:topn]
            return (associations)


class SP10K_Eval:
    """Basic-usage
    ------
    from spvec_eval import SP_Eval
    Eval = SP_Eval()
    Eval.load_model_files(<file-path>)

    # load a single model-file to load multiple files and more options see
    # docstring

    Eval.evaluate()

    """

    def __init__(self, sp10k_path='SP-10K/data', exclude=['wino']):
        """
        Parameters
        -----
        sp10k_path : str, default='SP-10K/data'
            the path to the directory that contains SP-10K dataset files

        exclude : list of str, default=['wino']
            specify the sp10k categories to exclude
            use this to exclude winograd files
        """

        self.param_text = {
            'ws': 'window-size',
            'dim': 'dimension',
            'tw': 'term-weight',
            'p': 'exponential-weight',
            'min': 'min-count'
        }

        self.modelfiles = {}
        self.modelresults = {}
        self.sp_dataset = self.load_sp10k(sp10k_path, exclude)

    def load_sp10k(self, path, exclude):
        """
        """
        sp10k_data = {}
        file_names = glob(str(Path(path, '*annotation*')))

        for file_name in file_names:
            cat = Path(file_name).parts[-1].replace("_annotation.txt", '')
            # sp-10k categories: 'dobj','amod','nsubj','nsubj_amod','dobj_amod',
            # 'wino_dobj_amod', 'wino_nsubj_amod'
            if any([exclude_str in cat for exclude_str in exclude]):
                # exclude any category
                # basically to exclude the winograd files
                continue

            items = []
            with open(file_name) as file_:
                for line in file_:
                    word1, word2, score = line.split()
                    items.append((word1, word2, float(score)))
                sp10k_data[cat] = items

        return sp10k_data

    def load_model_files(self, model_path, model_class='spvec', index_params=None,
                         having_params=None):
        """
        Add one or many model-files.
        This can be called multiple times to add models of different classes

        Parameters
        -----
        model_path :
            the path to a single model file or to directory with many models
            If directory, looks for files that end with .vec filename

        model_class : str, default=spvec
            some name to group a set of models in one directory, eg. word2vec, fasttext, etc


        ---following parameters are useful when you have model parameter-values encoded in filenames

        index_params : list of str
            important parameters that can index the model, eg. ws for window-size
            this allows to study the effect of these params on the SP-10K

        having_params: list, default=None
            keep only model files having specified parameters/string, eg. 'tw:ppmi'
        """

        if index_params is None:
            index_params = []

        if having_params is None:
            having_params = []

        if 'mt' not in index_params:
            # this is always required to differentiate syntagmatic and
            # paradigmatic embeddings
            index_params.append('mt')

        if Path(model_path).is_dir():
            model_filenames = glob(str(Path(model_path, '*.vec')))
        else:
            model_filenames = [model_path]


        if model_class not in self.modelfiles.keys():
            # check if model class exists
            self.modelfiles[model_class] = []
        else:
            print("model-class {} exists, replacing...".format(model_class))
            self.modelfiles[model_class] = []


        for filename in model_filenames:
            if all([param_str in filename for param_str in having_params]):
                # keep only the selected files
                params = {}
                for paramname in index_params:
                    regex = re.compile(paramname+":(.+?)_")

                    try:
                        paramvalue = regex.search(filename).group(1)
                        paramvalue = str2num_if_possible(paramvalue)
                    except AttributeError:
                        if paramname == 'mt':
                            paramvalue = 'par'
                            # assume paradigmatic
                        else:
                            raise AttributeError("Cant read parameter-value from filename,\
                                    see filename format")
                    params[paramname] = paramvalue

                self.modelfiles[model_class].append((params, filename))



    def evaluate(self, categories=['amod', 'dobj', 'nsubj']):
        """
        compute correlation with every category in sp10k with all models

        Parameters
        ----------
        categories : list of str, default=[amod,dobj,nsubj]
            evaluate only these categories in SP10K
        """
        assert len(self.modelfiles) > 0, "no model-files added, nothing to evaluate"

        Result = namedtuple('Result', ['indexes', 'correlation', 'significance'])
        tmp = []

        for model_class in self.modelfiles:
            print(model_class+"________")

            for indexes, filename in self.modelfiles[model_class]:
                model_type = indexes['mt']
                print(filename)
                model = SP_Model(filename, model_type)

                oovs = set()
                results = []
                for cat in self.sp_dataset.keys():
                    if cat not in categories:
                        continue
                    model_scores = []
                    # scores from model
                    gt_scores = []
                    # ground truth scores

                    for item in self.sp_dataset[cat]:
                        word1, word2, gt_score = item
                        word1 = word1.lower()
                        word2 = word2.lower()
                        if not model.in_vocab(word1):
                            oovs.add(word1)
                            continue
                        if not model.in_vocab(word2):
                            oovs.add(word2)
                            continue
                        model_scores.append(model.score(word1, word2, cat))
                        gt_scores.append(gt_score)


                    correlation, significance = spearmanr(model_scores, gt_scores)
                    print("\t"+cat+":",
                          "correlation=", round(correlation, 3),
                          "pvalue=", round(significance, 3),
                          "oovs=", len(oovs))

                    tmp.append(correlation)
                    result = Result(indexes, correlation, significance)
                    results.append((cat, result))
                print("\tAVG CORR:", round(sum(tmp)/len(tmp), 3))

                if model_class not in self.modelresults:
                    self.modelresults[model_class] = {}

                for cat, result in results:
                    if cat not in self.modelresults[model_class]:
                        self.modelresults[model_class][cat] = []
                    self.modelresults[model_class][cat].append(result)


    def plot_variations(self, x_axis_index, model_class='all', averaged=False,
                        plot_title=None, ignore_cat=None):
        """
        Plot the variation in correlation with different models.
        This expects each model to vary in some parameters.
        For example if the model vary in window-size,
        this can be use to plot the corresponding correlations.

        Model-files should be loaded with appropriate index_parameters
        then the x-axis will show the variation in parameter-value,
        and y axis the corresponding correlation

        Parameters
        ------
        x_axis_index : str,
            the index parameter for the x-axis

        model_class : str, default=all
            the same name given during `load_model_files`

        averaged : bool, default=False
            Plot average correlation over all SP-10K categories

        plot_title : str, default=None
            if None the title is "SP-10K Evaluation on "+model_class

        ignore_cat : list, default=None
            specify categories to ignore, eg. ['dobj_amod','nsubj_amod']
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
            print(model_class_)
            xs = []
            for cat, results in self.modelresults[model_class_].items():
                if cat in ignore_cat:
                    continue
                xs = []
                ys = []
                for result in results:
                    ys.append(result.correlation)
                    xs.append(result.indexes[x_axis_index])
                xs_ys = list(zip(xs, ys))
                xs_ys = sorted(xs_ys, key=lambda xy: xy[0])
                xs, ys = zip(*xs_ys)
                if averaged is False:
                    plt.plot(xs, ys, label=model_class_+":"+cat, marker='o')
                else:
                    ys_all.append(ys)

            if averaged:
                ys_cat_avg = [sum([row[i] for row in ys_all]) for i in range(len(ys_all[0]))]
                ys_cat_avg = [sum_/len(ys_all) for sum_ in ys_cat_avg]
                plt.plot(xs, ys_cat_avg, label=model_class_, marker='o')

        plt.xlabel(self.param_text[x_axis_index])
        plt.ylabel("spearmans-correlation")

        if model_class == 'all':
            model_str = ','.join(model_classes)
        else:
            model_str = model_classes[0]

        if plot_title is None:
            plt.title("SP-10K Spearmans Correlation on " + model_str +
                      "\nwith varying " + self.param_text[x_axis_index])
        else:
            plt.title(plot_title)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


def str2num_if_possible(str):
    """
    If possible converts string to number
    else return the same
    """
    try:
        return int(str)
    except ValueError:
        try:
            return float(str)
        except ValueError:
            return str


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
    we treat both left and right embedding of a word as separate words.
    By suffixing '_l' and '_r' to the corresponding word and embedding.
    Now gensim can seen all the embeddings in the same space, and do its magic.

    """

    def __init__(self, emb_file, emb_type='syn'):
        """
        """
        self.model = KeyedVectors.load_word2vec_format(emb_file, binary=False)
        self.model_type = emb_type

    def in_vocab(self, word):
        if self.model_type == 'syn':
            word = word+'_l'

        if word in self.model.index_to_key:
            return True
        else:
            return False

    def score(self, word1, word2, cat=None):
        if self.model_type == "syn":
            return self.syn_score(word1, word2, cat)
        elif self.model_type == "par":
            return self.par_score(word1, word2)
        else:
            raise ValueError("Unknown model_type {}".format(self.model_type))

    def par_score(self, word1, word2):
        """Return Syntagmatic score of two words.
        """
        score = 1 - self.model.distance(word1, word2)
        return score


    def syn_score(self, word1: str, word2: str, cat: str):
        """Return Syntagmatic score of two words according to category.

        Args:
            word1: word1 is the base word
            word2: word2 is either an amod, nsubj or dobj
            cat: One of the supported categies [amod, nsubj, dobj, nsubj_amod, dobj_amod]
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
            raise ValueError("Unkonwn category {}".format(cat))
        score = 1 - self.model.distance(word1, word2)
        return score
