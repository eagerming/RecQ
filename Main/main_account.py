import os
import sys

# from tensorflow.python.estimator import keras


relative_path_root = '..'
print(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
# sys.path.append(relative_path_root)

from data.account_data import AccountDAO
from Main.RecQ_multi_algorithums import RecQMultiAlgo


import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras




from tool.config import Config

if __name__ == '__main__':



    # order = '9'
    # # order = '10'
    # # order = '11'
    # # order = 'a1'
    # order = 'd4'
    # order = 'd1'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    import time
    s = time.time()

    algorthms = {'1':'UserKNN','2':'ItemKNN','3':'BasicMF','4':'SlopeOne','5':'SVD','6':'PMF',
                 '7':'SVD++','8':'EE','9':'BPR','10':'WRMF','11':'ExpoMF',
                 's1':'RSTE','s2':'SoRec','s3':'SoReg','s4':'SocialMF','s5':'SBPR','s6':'SREE',
                 's7':'LOCABAL','s8':'SocialFD','s9':'TBPR','s10':'SEREC','a1':'CoFactor',
                 'a2':'CUNE_MF','a3':'CUNE_BPR','a4':'IF_BPR',
                 'd1':'APR','d2':'CDAE','d3':'DMF','d4':'NeuMF','d5':'CFGAN','d6':'IRGAN','d7':'SRGAN',
                 'b1':'UserMean','b2':'ItemMean','b3':'MostPopular','b4':'Rand',
                 'ABPR':'ABPR'}


    algorithm_list = ['ABPR', 'BPR', 'WRMF', 'ExpoMF', 'CoFactor']
    algorithm_list = ['NeuMF', 'APR']
    # algorithm_list = ['ABPR']

    config_dict = {name: Config(os.path.join(relative_path_root, 'config', name + '.conf')) for name in algorithm_list}

    conf_account = Config('../config/account_setting.conf')
    account_data = AccountDAO(conf_account)

    for name, config in config_dict.items():
        config.update_inferior(conf_account)

    recSys_all = RecQMultiAlgo(config_dict,conf_account,account_data)
    recSys_all.execute_all(debug=True)

    # conf = Config('../config/'+algorthms[order]+'.conf')
    # conf.update_inferior(conf_account)
    # recSys = RecQ(conf, account_data)
    # recSys.execute()
    e = time.time()
    print("Run time: %f s" % (e - s))
