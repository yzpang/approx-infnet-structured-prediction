import argparse
import pprint
import sys


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train', type=str, default='')
    argparser.add_argument('--dev', type=str, default='')
    argparser.add_argument('--test', type=str, default='')
    argparser.add_argument('--embedding', type=str, default='')
    argparser.add_argument('--output', type=str, default='')

    argparser.add_argument('--load_model', type=str2bool, default=False)
    argparser.add_argument('--model_path', type=str, default='')


    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--max_epochs', type=int, default=150)
    argparser.add_argument('--dropout', type=float, default=0.6) # keep prob

    argparser.add_argument('--dim_h', type=int, default=128)
    argparser.add_argument('--dim_emb', type=int, default=100)
    argparser.add_argument('--learning_rate', type=float, default=0.0003)

    argparser.add_argument('--steps_per_checkpoint', type=int, default=3000)
    argparser.add_argument('--num_theta_steps', type=int, default=5)

    argparser.add_argument('--lambda', type=float, default=1.0)   



    args = argparser.parse_args()


    print('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('------------------------------------------------')

    return args