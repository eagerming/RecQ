
from algorithm.ranking.ABPR import ABPR
from tool.config import Config, LineConfig
from tool.file import FileIO
from evaluation.dataSplit import *
from multiprocessing import Process, Manager, cpu_count, Pool
from tool.file import FileIO
from time import strftime, localtime, time


class RecQMultiAlgo(object):
    def __init__(self, config_dict, config_account, account_DAO):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.relation = []
        self.measure = []
        self.config_dict = config_dict

        self.accountDAO = account_DAO

        if config_account.contains('evaluation.setup'):
            all_evaluation = LineConfig(config_account['evaluation.setup'])
            if all_evaluation.contains('--account'):
                self.training_user_item = account_DAO.training_user_item
                self.training_account_item = account_DAO.training_account_item
                self.relation = account_DAO.relation
                self.test_user_item = account_DAO.test_user_item
        else:
            raise Exception('Evaluation is not well configured!')

        print('preprocessing...')


    def execute_all(self, debug=False):

        if debug:
            for name, config in self.config_dict.items():
                self.execute(config)
        else:
            pool = Pool(cpu_count())

            for name, config in self.config_dict.items():
                pool.apply_async(self.execute, args=(config,))

            pool.close()
            pool.join()


    def execute(self, config, max_sample=1000):
        # import the algorithm module

        importStr = 'from algorithm.ranking.' + config['recommender'] + ' import ' + config['recommender']
        exec(importStr)


        algo_evaluation = LineConfig(config['evaluation.setup'])
        if algo_evaluation.contains('-ul') and eval(algo_evaluation['-ul']) > 0:
            training_data = 'self.training_user_item'
            social_info = 'relation=self.relation'
        else:
            training_data = 'self.training_account_item'
            social_info = ''

        recommender = config['recommender'] + '(config, {}, self.test_user_item, {})'.\
            format(training_data, social_info)

        algorithum = eval(recommender)
        algorithum.accountDAO = self.accountDAO
        algorithum.evaluation_conf = algo_evaluation
        algorithum.get_test_map()
        algorithum.get_test_sample_data(max_sample=max_sample)

        algorithum.execute()



