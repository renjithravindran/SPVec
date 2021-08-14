"""
SPVec: Syntagmatic Word Embeddings for
Unsupervised Learning of Selectional Preferences
-----
renjith p ravindran
akshay badola
-----
2021
-------------------------------------------------
This software distribution allows to 
[1] learn syntagmatic word embeddings
[2] evaluate the embeddings on the SP-10K dataset (for selectional preference)
[3] query the left and right associations of words

SPVec gives two embeddings per word, one of left context and other of right context.
If l_w is the left embedding of a word w and r_w the right embedding,
then the association between word v to the right of word u is given by cosine(r_u,l_v)

Look into the spvec library for more flexibility in learning and evaluating the embeddings.
-------------------------------------------------
[1] Renjith P. Ravindran, Akshay Badola, and Narayana Murthy. Syntagmatic
word embeddings for unsupervised learning of selectional preferences. In
Proceedings of the 6th Workshop on Representation Learning for NLP, On-
line, Aug 2021. Association for Computational Linguistics.
"""

import sys
import argparse
from pathlib import Path
from multiprocessing import cpu_count
import textwrap


def formatter(prog):
    return argparse.RawTextHelpFormatter(prog,width=99999,max_help_position=30)

def learn(arglist):
    """
    """

    parser = argparse.ArgumentParser(description=desc, allow_abbrev=False, add_help=False, formatter_class=formatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    help_str = 'path to a single file pre-processed corpus'

    requiredNamed.add_argument('--corpus-file', required=True, metavar='<CORP>', help=help_str)

    optionalNamed = parser.add_argument_group('optional named arguments')
    help_str = 'available=syn|par (default=syn)'
    optionalNamed.add_argument('--model-type', default="syn", 
                            metavar='<TYPE>', choices=['syn', 'par'], help=help_str)

    help_str = '(default=3)'
    optionalNamed.add_argument('--window-size', metavar='<WIN>', default='3', type=int, help=help_str, required=False)

    help_str = '(default=300)'
    optionalNamed.add_argument('--dimensions', metavar='<DIM>', default='300', help=help_str, type=int, required=False) 

    help_str = 'available=raw|log|pmi|ppmi (default=log)' 
    choices = ['raw', 'log', 'pmi', 'ppmi']
    optionalNamed.add_argument('--term-weight', metavar='<TERM>', choices=choices, help=help_str, default='log') 
    
    help_str = '(default=0.5)'
    optionalNamed.add_argument('--power-factor', metavar='<POW>', default='0.5', help=help_str, type=float, required=False) 

    help_str = 'no. of CPUs for building the co-occurrence matrix (default={})'.format(int(cpu_count()/2))
    optionalNamed.add_argument('--jobs', metavar='<JOBS>', type=int, help=help_str, default=int(cpu_count()/2), required=False)


    help_str = 'prefix string for embedding filename (default=spvec)'
    optionalNamed.add_argument('--model-prefix', metavar='<PRE>', help=help_str, default='spvec', required=False) 

    help_str = """\
                either a directory or a full filename. If directory the filename
                will be formatted from model parameters (default=./)
                """
    help_str = textwrap.dedent(help_str)
    optionalNamed.add_argument( '--save-to', metavar='<FILE>', default='./', help=help_str, required=False)

    optionalSwitches = parser.add_argument_group('optional switches')
    help_str = 'sparse factorisation in-case factorisation runs out of memory'
    optionalSwitches.add_argument('--sparse', help=help_str, action='store_true', default=False, required=False)
    
    if "-h" in arglist or "--help" in arglist or '--h' in arglist:
        parser.print_help()
        return 0
    args = parser.parse_args(arglist)

    from lib.spvec import SPVec
    spvec = SPVec(corpus_filename=args.corpus_file,
                  modeltype=args.model_type,
                  windowsize=args.window_size,
                  jobs=args.jobs,
                  model_prefix=args.model_prefix)
    spvec.make_embeddings(term_weight=args.term_weight,
                          dim=args.dimensions,
                          p=args.power_factor,
                          save_to=args.save_to,
                          sparse=args.sparse)


def eval(arglist):
    """
    """

    desc = "SPVec Evaluation on SP-10k"
    epilog = """\
            NOTE: This can be used to evaluate any embeddings which are in word2vec format.
            Any embedding file without 'mt:syn' in filename is considered paradigmatic.
            """
    epilog = textwrap.dedent(epilog)
    parser = argparse.ArgumentParser(description=desc,
                                     usage='python %(prog)s eval --model-file <EMBEDDINGS.VEC>',
                                     epilog=epilog,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)

    requiredNamed = parser.add_argument_group('required named arguments')
    help_str = """\
                word embedding file, or directory with many embedding files.
                embedding files should have .vec as extension.
                spvec embeddings should have the model parameters in filename.
                """
    help_str = textwrap.dedent(help_str)
    requiredNamed.add_argument("--model-file", metavar='<MODEL>', required=True, help=help_str)

    optionalNamed = parser.add_argument_group('optional named arguments')
    help_str = "path to directory containing the SP-10K dataset (default=./SP-10K/data)"
    optionalNamed.add_argument("--data-set", metavar='<DSET>', default="SP-10K/data", help=help_str)

    if any([item in arglist for item in ['-h','--h','--help']]):
        parser.print_help()
        return 0

    args = parser.parse_args(arglist)

    data_path = Path(args.data_set)
    if not data_path.exists() or not data_path.is_dir():
        raise AttributeError("Data path must exist and must be a directory")
    from lib.spvec_eval import SP10K_Eval
    sp_eval = SP10K_Eval(data_path)
    print("\nEvaluating...")
    sp_eval.load_model_files(args.model_file)
    sp_eval.evaluate()


def query(arglist):
    """
    """

    desc = "SPVec Query for left/right Word Associations"
    parser = argparse.ArgumentParser(description=desc,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--model-file", metavar='<MODEL>',
                        required=True,
                        help="word embedding file")
    if any([item in arglist for item in ['-h','--h','--help']]):
        parser.print_help()
        return 0
    args = parser.parse_args(arglist)
    print("\nLoading model. Please wait...")
    
    from lib.spvec_eval import SP_Query
    sp_query = SP_Query(args.model_file)

    def valid(inp):
        return any(map(bool, [len(inp)]))

    prompt = "_________\nEnter any word to print the associations (enter \"q\" to quit): "
    while True:
        x = input(prompt).strip()
        if x == "q":
            return 0
        elif not valid(x):
            print(f"Invalid input {x}")
        else:
            sp_query.get_associations(x)


if __name__ == '__main__':
    desc = """\
            SPVec: Syntagmatic Word Embeddings
            ------
            """
    desc = textwrap.dedent(desc)
    epilog = """\
                ---------------------------------
                Type "python spvec.py command --help" to get help about the individual commands
                ---------------------------------
            """
    usage = "\"python spvec.py learn --corpus-file <CORPUS.TXT>\""
    epilog = textwrap.dedent(epilog)
    parser = argparse.ArgumentParser(description=desc,
                                     epilog=epilog,
                                     usage=usage,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)
    requiredArgs = parser.add_argument_group('required arguments')
    help_str = """\
                    available=learn|eval|query
                """
    requiredArgs.add_argument("command", help=help_str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    elif sys.argv[1] in {"-h", "--help","--h"}:
        parser.print_help()
        sys.exit(0)
    try:
        args, sub_args = parser.parse_known_args()
    except Exception:
        parser.print_help()
        sys.exit(1)
    if args.command == "learn":
        learn(sub_args)
    elif args.command == "eval":
        eval(sub_args)
    elif args.command == "query":
        query(sub_args)
    else:
        print(f"Unknown command \"{args.command}\"\n")
        parser.print_help()
        sys.exit(1)
