#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
import gensim.models.doc2vec as d2v
from tool.qmath import cosine
from gensim.models.doc2vec import TaggedDocument
import pickle
from tool import config


class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename =filename
    def __iter__(self):
        with open(self.filename) as f:
            self.label = []
            self.abstracts = []
            for num, line in enumerate(f):
                if num % 2 == 0:
                    self.label.append(line.strip()[2:])
                if num % 2 == 1:
                    self.abstracts.append(line.strip().split(' '))
        self.sentences = zip(self.abstracts, self.label)
        for line in self.sentences:
            yield TaggedDocument(words=line[0], tags=[line[1]])

class PaperRec(IterativeRecommender):


    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(PaperRec, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(PaperRec, self).readConfiguration()
        options = config.LineConfig(self.config['PaperRec'])
        self.topK = int(options['-topK'])
        self.beta = float(options['-beta'])

    def initModel(self):
        super(PaperRec, self).initModel()
        self.Z = np.random.rand(self.dao.trainingSize()[1], self.k)
        # self.sentences = LabeledLineSentence('../dataset/citeulike/rawtext.dat')
        # model = d2v.Doc2Vec(self.sentences,size=100, window=8, min_count=3, workers=4,iter=10)
        # self.W = np.random.rand(self.dao.trainingSize()[1], 100)
        # for item in self.dao.item:
        #     iid = self.dao.item[item]
        #     self.W[iid] = model.docvecs[item]
        # print 'Constructing similarity matrix...'
        # i = 0
        # self.topKSim = {}        #
        # for item in self.dao.item:
        #     iSim = []
        #     i += 1
        #     if i % 200 == 0:
        #         print i, '/', len(self.dao.item)
        #     id1 = self.dao.item[item]
        #     vec1 = self.W[id1]
        #     for item2 in self.dao.item:
        #         if item <> item2:
        #             id2 = self.dao.item[item2]
        #             vec2 = self.W[id2]
        #             sim = cosine(vec1, vec2)
        #             iSim.append((item2, sim))
        #
        #     self.topKSim[item] = sorted(iSim, key=lambda d: d[1], reverse=True)[:self.topK]
        #
        ## save similarity to disk
        # output = open('similarity.pkl', 'wb')
        # # Pickle dictionary using protocol 0.
        # pickle.dump(self.topKSim, output)

        #load similarity from disk
        f = open('similarity.pkl','rb')
        self.topKSim = pickle.load(f)



    def buildModel(self):

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        #self.NegativeSet = defaultdict(list)

        for user in self.dao.user:
            for item in self.dao.trainSet_u[user]:
                if self.dao.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1
                # else:
                #     self.NegativeSet[user].append(item)
        print 'training...'
        iteration = 0
        itemList = self.dao.item.keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.PositiveSet:
                u = self.dao.user[user]
                for item in self.PositiveSet[user]:
                    i = self.dao.item[item]
                    # if len(self.NegativeSet[user]) > 0:
                    #     item_j = choice(self.NegativeSet[user])
                    # else:
                    item_j = choice(itemList)
                    while (self.PositiveSet[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.dao.item[item_j]
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.lRate * (1 - s) * self.P[u]
                    self.Q[j] -= self.lRate * (1 - s) * self.P[u]

                    self.P[u] -= self.lRate * self.regU * self.P[u]
                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.Q[j] -= self.lRate * self.regI * self.Q[j]
                    self.loss += -log(s)

            for item in self.topKSim:
                if not self.dao.item.has_key(item):
                        continue
                id1 = self.dao.item[item]
                for similarItem in self.topKSim[item]:
                    if not self.dao.item.has_key(similarItem[0]):
                        continue
                    id2 = self.dao.item[similarItem[0]]
                    p = self.Q[id1]
                    z = self.Z[id2]
                    error = similarItem[1] -p.dot(z)
                    self.Q[id1]+=self.beta*self.lRate*error*z
                    self.Z[id2]+=self.beta*self.lRate*error*p
                    self.Z[id2]-=self.beta*self.lRate*self.regI*z
                    self.loss+=self.beta*error**2
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()+self.regI * (self.Z * self.Z).sum()
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self,user,item):

        if self.dao.containsUser(user) and self.dao.containsItem(item):
            u = self.dao.getUserId(user)
            i = self.dao.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return sigmoid(self.dao.globalMean)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.dao.containsUser(u):
            u = self.dao.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.dao.globalMean] * len(self.dao.item)


