import sys
import argparse
from pathlib import Path
from multiprocessing import cpu_count


def formatter(prog):
    return argparse.RawTextHelpFormatter(prog,
                                         width=99999,
                                         max_help_position=27)


def train(arglist):
    save_to_help = """either a directory or a full filename. if directory the fileame
will be formatted from model parameters (default=./)
"""
    parser = argparse.ArgumentParser(description=desc,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)
    parser.add_argument('-c', '--corpus-file',
                        required=True,
                        metavar='<corpus-file>',
                        help='Path to single file pre-processed corpus')
    parser.add_argument('-mt', '--model-type',
                        default="syn",
                        metavar='<model-type>',
                        choices=['syn', 'par'],
                        help='Available=syn|par')
    parser.add_argument(
        # '-w',
        '--window-size',
        metavar='WIN',
        default='3',
        type=int,
        help='(default=3)',
        required=False)
    parser.add_argument(
        # '-d',
        '--dimensions',
        metavar='DIM',
        default='300',
        help='(default=300)',
        type=int,
        required=False)
    parser.add_argument(
        # '-t',
        '--term-weight',
        metavar='TERM',
        choices=['raw', 'log', 'pmi', 'ppmi'],
        help='available=raw|log|pmi|ppmi (default=log)',
        default='log')
    parser.add_argument(
        # '-p',
        '--power-factor',
        metavar='POW',
        default='0.5',
        help='(default=0.5)',
        type=float,
        required=False)
    parser.add_argument(
        # '-j',
        '--jobs',
        metavar='JOBS',
        type=int,
        help='no. of CPUs for building the co-occurrence matrix (default={})'.format(
            int(cpu_count()/2)),
        default=int(cpu_count()/2),
        required=False)
    parser.add_argument(
        # '-u',
        '--sparse',
        help='sparse factorisation in-case factorisation runs out of memeory',
        action='store_true',
        default=False,
        required=False)
    parser.add_argument(
        # '-m',
        '--model-prefix',
        metavar='PRE',
        help='prefix string for embedding filename (default=spvec)',
        default='spvec',
        required=False)
    parser.add_argument(
        # '-s',
        '--save-to',
        metavar='FILE',
        default='./',
        help=save_to_help,
        required=False)

    if "-h" in arglist or "--help" in arglist:
        parser.print_help()
        return 0
    args = parser.parse_args(arglist)

    from lib.spvec import SPVec
    spvec = SPVec(corpus_filename=args.corpus_filename,
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
    parser = argparse.ArgumentParser(description=desc,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)
    parser.add_argument("-m", "--model-file", metavar='MODEL',
                        required=True,
                        help="Path for the trained embeddings")
    parser.add_argument("-d", "--data-path", metavar='DATA',
                        default="SP-10K/data",
                        help="Path to directory containing the SP-10K dataset")
    if "-h" in arglist or "--help" in arglist:
        parser.print_help()
        return 0
    args = parser.parse_args(arglist)
    data_path = Path(args.data_path)
    if not data_path.exists() or not data_path.is_dir():
        raise AttributeError("Data path must exist and must be a directory")
    from lib.spvec_eval import SP10K_Eval
    sp_eval = SP10K_Eval(data_path)
    print("\nEvaluating...")
    sp_eval.load_model_files(args.model_file)
    sp_eval.evaluate()


def query(arglist):
    parser = argparse.ArgumentParser(description=desc,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)
    parser.add_argument("-m", "--model-file", metavar='MODEL',
                        required=True,
                        help="Path for the trained embeddings")
    if "-h" in arglist or "--help" in arglist:
        parser.print_help()
        return 0
    args = parser.parse_args(arglist)
    from lib.spvec_eval import SP_Query
    print("\nLoading model. Please wait...")
    sp_query = SP_Query(args.model_file)

    def valid(inp):
        return any(map(bool, [len(inp)]))

    prompt = "Enter any word to print the associations (enter \"q\" to quit): "
    while True:
        x = input(prompt).strip()
        if x == "q":
            return 0
        elif not valid(x):
            print(f"Invalid input {x}")
        else:
            sp_query.get_associations(x)


if __name__ == '__main__':
    cmd_list = ["train", "eval", "query"]
    desc = """SPVec: Syntagmtic and Paradigmatic Word Embeddings
from word co-occurrence matrix decompostion.
------
renjith p ravindran 2021
"""

    usage = "Usage \"python main.py command [options]\""
    parser = argparse.ArgumentParser(description=desc,
                                     allow_abbrev=False,
                                     add_help=False,
                                     formatter_class=formatter)
    parser.add_argument("command", help=f"""Command to run.

command is one of {cmd_list}

Type "python main.py command --help" to get help about the individual commands.""")
    if len(sys.argv) == 1:
        print(f"No command given\n")
        parser.print_help()
        sys.exit(1)
    elif sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        sys.exit(0)
    try:
        args, sub_args = parser.parse_known_args()
    except Exception:
        parser.print_help()
        sys.exit(1)
    if args.command == "train":
        train(sub_args)
    elif args.command == "eval":
        eval(sub_args)
    elif args.command == "query":
        query(sub_args)
    else:
        print(f"Unknown command \"{args.command}\"\n")
        parser.print_help()
        sys.exit(1)
