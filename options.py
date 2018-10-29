import argparse
import pprint
import sys




def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train',
            type=str,
            default='')
    argparser.add_argument('--dev',
            type=str,
            default='')
    argparser.add_argument('--test',
            type=str,
            default='')
    argparser.add_argument('--embedding',
            type=str,
            default='')
    argparser.add_argument('--output',
            type=str,
            default='')

    argparser.add_argument('--model',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            type=str,
            default='False')

    argparser.add_argument('--batch_size',
            type=int,
            default=1)
    argparser.add_argument('--max_epochs',
            type=int,
            default=200)
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=200)

    argparser.add_argument('--dim_h',
            type=int,
            default=100)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)

    argparser.add_argument('--dropout', # keep prob
            type=float,
            default=0.6)

    args = argparser.parse_args()

    print('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('------------------------------------------------')

    return args