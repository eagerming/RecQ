# Copyright (C) 2016 School of Software Engineering, Chongqing University
#
# This file is part of RecQ.
#
# RecQ is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from collections import OrderedDict
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

from joblib import Parallel, delayed
from tqdm import tqdm

from data.rating import RatingDAO
from tool.file import FileIO
from tool.qmath import denormalize
from tool.config import Config, LineConfig
from os.path import abspath
from time import strftime, localtime, time
from evaluation.measure import Measure
import numpy as np


class Recommender(object):
    def __init__(self, conf, trainingSet, testSet, fold='[1]'):
        self.currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True

        self.data = RatingDAO(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.measure = []
        self.record = []
        if self.evalSettings.contains('-cold'):
            # evaluation on cold-start users
            threshold = int(self.evalSettings['-cold'])
            removedUser = {}
            for user in self.data.testSet_u:
                if user in self.data.trainSet_u and len(self.data.trainSet_u[user]) > threshold:
                    removedUser[user] = 1

            for user in removedUser:
                del self.data.testSet_u[user]

            testData = []
            for item in self.data.testData:
                if item[0] not in removedUser:
                    testData.append(item)
            self.data.testData = testData

        self.num_users, self.num_items, self.train_size = self.data.trainingSize()

    def get_test_sample_data(self, max_sample=1000):

        testSample = {}
        keys = list(self.data.testSet_u.keys())
        if len(self.data.testSet_u) <= max_sample:
            testSample = self.data.testSet_u
        else:
            while True:
                if len(testSample) == max_sample:
                    break
                index = np.random.choice(len(self.data.testSet_u))
                user = keys[index]
                testSample[user] = self.data.testSet_u[user]

        self.testSample = testSample

    def get_test_map(self, K=1, L=-1):
        self.K = K
        self.L = L
        if not hasattr(self, 'accountDAO') or self.accountDAO is None:
            self.map_from_true_to_identify = {i: i for i in list(self.data.testSet_u.keys())}
        elif self.evaluation_conf.contains('-ul') and eval(self.evaluation_conf['-ul']) > 0:
            self.map_from_true_to_identify = self.get_map_from_true_to_identify(k=K, index=L)
        else:
            self.map_from_true_to_identify = self.accountDAO.map_from_user_to_account

    def readConfiguration(self):
        self.algorName = self.config['recommender']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        print('Algorithm:', self.config['recommender'])
        print('Ratings dataset:', abspath(self.config['ratings']))
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(LineConfig(self.config['evaluation.setup']).getOption('-testSet')))
        # print 'Count of the users in training set: ',len()
        print('Training set size: (user count: %d, item count %d, record count: %d)' % (self.data.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' % (self.data.testSize()))
        print('=' * 80)

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def buildModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self, u, i):
        pass

    def predictForRanking(self, u):
        pass

    def checkRatingBoundary(self, prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction, 3)

    def evalRatings(self):
        res = []  # used to contain the text of the result
        res.append('userId  itemId  original  prediction\n')
        # predict
        for ind, entry in enumerate(self.data.testData):
            user, item, rating = entry

            # predict
            prediction = self.predict(user, item)
            # denormalize
            # prediction = denormalize(prediction,self.data.rScale[-1],self.data.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.data.testData[ind].append(pred)
            res.append(user + ' ' + item + ' ' + str(rating) + ' ' + str(pred) + '\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender'] + '@' + currentTime + '-rating-predictions' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

    def get_map_from_true_to_identify(self, k=1, index=-1):
        map_from_true_to_identify = {}
        table = self.accountDAO.test_table[self.accountDAO.test_table.k == k]

        reserve_list = [ind for ind, users in enumerate(table['identify_user']) if len(users)]
        table = table.iloc[reserve_list].copy()

        table['identify_user_index'] = [i_list[index] if len(i_list) and len(i_list) >= index + 1 else None for
                                        i_list in table['identify_user']]
        # table['identify_user_index'].astype(int)
        # table.groupby

        identify_list = table.groupby('truth_user')['identify_user_index'].aggregate(list)

        for truth, idens in identify_list.items():
            i_users, counts = np.unique(np.array(idens)[np.array(idens) > 0], return_counts=True)
            if len(i_users) == 0:
                continue
            map_from_true_to_identify[truth] = i_users[np.argmax(counts)]

        # identification_result = dict(zip(table['truth_user'].to_list(), table['identify_user'].to_list()))
        # for key, value in identification_result.items():
        #     if len(value) and len(value) >= index + 1:
        #         try:
        #             map_from_true_to_identify[key] = value[index]
        #         except:
        #             print(key, value)
        #             map_from_true_to_identify[key] = value[index]

        return map_from_true_to_identify

    def get_recommendation(self, data_user, N):
        user, identified_user, testSample_user = data_user
        itemSet = {}
        line = str(user) + ':'
        predictedItems = self.predictForRanking(identified_user)

        for id, rating in enumerate(predictedItems):
            itemSet[self.data.id2item[id]] = rating

        # if not hasattr(self, 'accountDAO') or self.accountDAO is None:
        #     ratedList, ratingList = self.data.userRated(user)
        # else:
        #     ratedList = list(self.accountDAO.ground_visit[user].keys())
        # for item in ratedList:
        #     del itemSet[item]

        Nrecommendations = []

        for item in itemSet:
            if len(Nrecommendations) < N:
                Nrecommendations.append((item, itemSet[item]))
            else:
                break

        # Nrecommendations = list(itemSet.items())[:N]

        Nrecommendations.sort(key=lambda d: d[1], reverse=True)
        recommendations = [item[1] for item in Nrecommendations]
        resNames = [item[0] for item in Nrecommendations]

        # find the N biggest scores
        for item in itemSet:
            ind = N
            l = 0
            r = N - 1

            if recommendations[r] < itemSet[item]:
                while r >= l:
                    mid = (r - l) // 2 + l
                    if recommendations[mid] >= itemSet[item]:
                        l = mid + 1
                    elif recommendations[mid] < itemSet[item]:
                        r = mid - 1

                    if r < l:
                        ind = r
                        break
            # move the items backwards
            if ind < N - 2:
                recommendations[ind + 2:] = recommendations[ind + 1:-1]
                resNames[ind + 2:] = resNames[ind + 1:-1]
            if ind < N - 1:
                recommendations[ind + 1] = itemSet[item]
                resNames[ind + 1] = item

        # recList[user] = list(zip(resNames, recommendations))

        # recList[user] = list(itemSet_sorted.items())[:N]

        recList_user = list(zip(resNames, recommendations))

        for item in recList_user:
            line += ' (' + str(item[0]) + ',' + str(item[1]) + ')'
            if item[0] in testSample_user:
                line += '*'

        line += '\n'

        return user, line, recList_user

    def evalRanking(self, write_to_file=True, use_now_time=False):
        res = []  # used to contain the text of the result

        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = max(top)
            if N > 100 or N < 0:
                print('N can not be larger than 100! It has been reassigned with 10')
                N = 10
            if N > len(self.data.item):
                N = len(self.data.item)
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)

        res.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        userN = {}

        testSample = self.testSample

        # # multiprocessing way
        # pool = Pool(12)
        # dataset = []
        # for user, testSample_u in testSample.items():
        #     identified_user = self.map_from_true_to_identify.get(user, -1)
        #     if identified_user == -1:
        #         continue
        #     dataset.append([user, identified_user, testSample_u])
        #
        # result_generator = pool.imap_unordered(partial(self.get_recommendation, N=N), dataset)
        # for result in tqdm(result_generator, total=len(dataset), desc='Measuring [{}]'):
        #     user, line, recList_user = result
        #     recList[user] = recList_user
        #     res.append(line)
        # pool.close()
        # pool.join()

        testSample_copy = testSample.copy()

        for i, user in tqdm(enumerate(testSample), total=len(testSample), desc='Measuring [{}]'.format(self.algorName)):
            identified_user = self.map_from_true_to_identify.get(user, -1)
            if identified_user == -1:
                del testSample_copy[user]
                continue
            user, line, recList_user = self.get_recommendation((user, identified_user, testSample[user]), N)

            recList[user] = recList_user
            res.append(line)

        self.measure = Measure.rankingMeasure(testSample_copy, recList, top)
        try:
            self.measure.append("C:{}\n".format(self.C))
        except:
            pass
        try:
            self.measure.append("L:{}\n".format(self.L))
        except:
            pass
        try:
            self.measure.append("K:{}\n".format(self.K))
        except:
            pass
        try:
            self.measure.append("N:{}\n".format(self.N))
        except:
            pass


        if use_now_time:
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        else:
            currentTime = self.currentTime
        if write_to_file:
            # output prediction result
            if False and self.isOutput:
                fileName = ''
                outDir = self.output['-dir']
                fileName = self.config['recommender'] + '@' + currentTime + '-top-' + str(
                    N) + 'items' + self.foldInfo + '.txt'
                FileIO.writeFile(outDir, fileName, res)
            # output evaluation result
            outDir = self.output['-dir']
            try:
                fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '_C{}'.format(self.C) + '.txt'
            except:
                fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, self.measure)
            # FileIO.writeFile(outDir, fileName, "C:{}".format(self.C))

            print('The result has been output to ', abspath(outDir), '.')
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

    def execute(self):
        self.readConfiguration()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        # load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' % (self.foldInfo))
            self.loadModel()
        else:
            print('Initializing model %s...' % (self.foldInfo))
            self.initModel()
            print('Building Model %s...' % (self.foldInfo))
            try:
                import tensorflow
                if self.evalSettings.contains('-tf'):
                    self.buildModel_tf()
                else:
                    self.buildModel()
            except ImportError:
                self.buildModel()

        # preict the ratings or item ranking
        print('Predicting %s...' % (self.foldInfo))
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()

        # save model
        if self.isSaveModel:
            print('Saving model %s...' % (self.foldInfo))
            self.saveModel()
        # with open(self.foldInfo+'measure.txt','w') as f:
        #     f.writelines(self.record)
        return self.measure
