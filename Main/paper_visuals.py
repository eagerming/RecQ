import argparse
import os
import sys

# from tensorflow.python.estimator import keras


relative_path_root = '..'
sys.path.append(os.path.abspath('..'))
# sys.path.append(relative_path_root)

from data.account_data import AccountDAO
from Main.RecQ_multi_algorithums import RecQMultiAlgo

# import tensorflow as tf
# from tensorflow.python.keras.backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras


from tool.config import Config


def parse_args(arg=None, namespace=None, parse_known_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', help="input the dataset among music, book and music")
    parser.add_argument('Param', help="varies with this parameter")
    # parser.add_argument('--movie', action='store_true', help="run movie data" )
    # parser.add_argument('--book', action='store_true', help="run book data")
    # parser.add_argument('--music', action='store_true', help="run music data")
    parser.add_argument('--debug', action='store_true', help="run in single processing if indicated")

    if parse_known_args:
        args, extra = parser.parse_known_args(arg, namespace=namespace)
        return args, extra
    else:
        args = parser.parse_args(arg, namespace=namespace)
        return args


def walk_dir(path):
    for fpathe, dirs, fs in os.walk(path):  # os.walk是获取所有的目录
        return dirs


def run_differ_L(args):
    pass


def run_differ_C(args):
    algorithm_list = ['ABPR', 'ABPR_10', 'ABPR_d', 'ABPR_sqrt']

    config_dict = {name: Config(os.path.join(relative_path_root, 'config', name + '.conf')) for name in algorithm_list}

    conf_path = os.path.join(relative_path_root, 'config', 'account_{}.conf'.format(args.dataset))
    print('Account configuration path: {}'.format(conf_path))

    conf_account = Config(conf_path)
    conf_account['output.setup'] = os.path.join(os.path.dirname(os.path.dirname(conf_account['output.setup'])),
                                                'varies_C/')

    account_data = AccountDAO(conf_account)

    for name, config in config_dict.items():
        if name.startswith('ABPR'):
            config.update_inferior(conf_account)
        else:
            config.update(conf_account)

    for C in range(1, 11):
        recSys_all = RecQMultiAlgo(config_dict, conf_account, account_data, C=C)
        # args.debug = True
        recSys_all.execute_all(debug=args.debug)


def run_differ_L(args):
    algorithm_list = ['ABPR', 'ABPR_10', 'ABPR_d', 'ABPR_sqrt', 'ABPR_t1', 'BPR', 'WRMF', 'ExpoMF', 'CoFactor',
                      'NeuMF', 'APR', 'CDAE', 'CFGAN']
    # algorithm_list = ['ABPR', 'ABPR_10', 'ABPR_d', 'ABPR_sqrt', 'ABPR_t1']

    config_dict = {name: Config(os.path.join(relative_path_root, 'config', name + '.conf')) for name in algorithm_list}

    conf_path = os.path.join(relative_path_root, 'config', 'account_{}.conf'.format(args.dataset))
    print('Account configuration path: {}'.format(conf_path))

    conf_account = Config(conf_path)
    conf_account['output.setup'] = os.path.join(os.path.dirname(os.path.dirname(conf_account['output.setup'])),
                                                'varies_LA/')

    account_data = AccountDAO(conf_account)

    for name, config in config_dict.items():
        if name.startswith('ABPR'):
            config.update_inferior(conf_account)
        else:
            config.update(conf_account)

    for L in range(0, 10):
        recSys_all = RecQMultiAlgo(config_dict, conf_account, account_data, L=L)
        # args.debug = True
        recSys_all.execute_all(debug=args.debug)


def run_differ_K(args):
    algorithm_list = ['ABPR', 'ABPR_10', 'ABPR_d', 'ABPR_sqrt', 'ABPR_t1', 'BPR', 'WRMF', 'ExpoMF', 'CoFactor',
                      'NeuMF', 'APR', 'CDAE', 'CFGAN']
    # algorithm_list = ['ABPR', 'ABPR_10', 'ABPR_d', 'ABPR_sqrt', 'ABPR_t1']

    config_dict = {name: Config(os.path.join(relative_path_root, 'config', name + '.conf')) for name in algorithm_list}

    conf_path = os.path.join(relative_path_root, 'config', 'account_{}.conf'.format(args.dataset))
    print('Account configuration path: {}'.format(conf_path))

    conf_account = Config(conf_path)
    conf_account['output.setup'] = os.path.join(os.path.dirname(os.path.dirname(conf_account['output.setup'])),
                                                'varies_KA/')

    account_data = AccountDAO(conf_account)

    for name, config in config_dict.items():
        if name.startswith('ABPR'):
            config.update_inferior(conf_account)
        else:
            config.update(conf_account)

    for K in range(1, 11):
        recSys_all = RecQMultiAlgo(config_dict, conf_account, account_data, K=K)
        # args.debug = True
        recSys_all.execute_all(debug=args.debug)


def run_differ_N(args):
    algorithm_list = ['ABPR', 'ABPR_10', 'ABPR_d', 'ABPR_sqrt', 'ABPR_t1']

    config_dict = {name: Config(os.path.join(relative_path_root, 'config', name + '.conf')) for name in algorithm_list}

    conf_path = os.path.join(relative_path_root, 'config', 'account_{}.conf'.format(args.dataset))
    print('Account configuration path: {}'.format(conf_path))

    conf_account = Config(conf_path)

    dirs = walk_dir('../account_data/lastfm-dataset-1K/differ_N')
    conf_account['account.data'] = os.path.join(os.path.dirname(conf_account['account.data']), 'differ_N/',
                                                os.path.basename(conf_account['account.data']))
    conf_account['output.setup'] = os.path.join(os.path.dirname(os.path.dirname(conf_account['output.setup'])),
                                                'varies_N/')

    for dir in dirs:
        conf_account['account.data'] = os.path.join(os.path.dirname(conf_account['account.data']), dir)
        # conf_account['output.setup'] = os.path.join(conf_account['output.setup'], dir)

        account_data = AccountDAO(conf_account)

        for name, config in config_dict.items():
            if name.startswith('ABPR'):
                config.update_inferior(conf_account)
            else:
                config.update(conf_account)

        recSys_all = RecQMultiAlgo(config_dict, conf_account, account_data, N=int(dir[1:]))

        args.debug = True
        recSys_all.execute_all(debug=args.debug)


if __name__ == '__main__':
    args, _ = parse_args(parse_known_args=True)
    args.debug = True
    if args.Param == 'C':
        run_differ_C(args)
    elif args.Param == 'L':
        run_differ_L(args)
    elif args.Param == 'K':
        run_differ_K(args)
    elif args.Param == 'N':
        run_differ_N(args)
