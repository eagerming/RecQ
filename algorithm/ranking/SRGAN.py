# coding:utf8
from baseclass.DeepRecommender import DeepRecommender
from baseclass.SocialRecommender import SocialRecommender
import numpy as np
from random import randint, choice,shuffle
from collections import defaultdict
import tensorflow as tf
import gensim.models.word2vec as w2v
from tool.qmath import sigmoid, cosine

def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(u + eps) + eps)
    y = tf.log(logits + eps) + gumbel_noise
    return tf.nn.softmax(y / temperature)



class SRGAN(SocialRecommender,DeepRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, fold='[1]'):
        DeepRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,fold=fold)

    def readNegativeFeedbacks(self):
        self.negative = defaultdict(list)
        self.nItems = defaultdict(list)
        filename = self.config['ratings'][:-4]+'_n.txt'
        with open(filename) as f:
            for line in f:
                items = line.strip().split()
                self.negative[items[0]].append(items[1])
                self.nItems[items[1]].append(items[0])
                if items[0] not in self.data.user:
                    self.data.user[items[0]]=len(self.data.user)
                    self.data.id2user[self.data.user[items[0]]] = items[0]
                    self.num_users+=1


    def randomWalks(self):

        self.positive = defaultdict(list)
        self.pItems = defaultdict(list)
        for user in self.data.trainSet_u:
            for item in self.data.trainSet_u[user]:
                self.positive[user].append(item)
                self.pItems[item].append(user)

        print('Kind Note: This method will probably take much time.')
        # build U-F-NET
        print('Building weighted user-friend network...')
        # filter isolated nodes and low ratings
        # Definition of Meta-Path
        p1 = 'UIU'
        p2 = 'UFU'
        p3 = 'UTU'
        p4 = 'UFIU'
        p5 = 'UFUIU'
        mPaths = [p1, p2, p3, p4, p5]

        self.walkLength=30
        self.topK = 100

        self.G = np.random.rand(self.num_users, 50) * 0.1
        self.W = np.random.rand(self.num_items, 50) * 0.1

        self.UFNet = defaultdict(list) # a -> b #a trusts b
        for u in self.social.followees:
            s1 = set(self.social.followees[u])
            for v in self.social.followees[u]:
                if v in self.social.followees:  # make sure that v has out links
                    if u != v:
                        s2 = set(self.social.followees[v])
                        weight = len(s1.intersection(s2))
                        self.UFNet[u] += [v] * (weight + 1)

        self.UTNet = defaultdict(list) # a <- b #a is trusted by b
        for u in self.social.followers:
            s1 = set(self.social.followers[u])
            for v in self.social.followers[u]:
                if v in self.social.followers:  # make sure that v has out links
                    if u != v:
                        s2 = set(self.social.followers[v])
                        weight = len(s1.intersection(s2))
                        self.UTNet[u] += [v] * (weight + 1)

        print('Generating random meta-path random walks... (Positive)')
        self.pWalks = []
        # self.usercovered = {}

        # positive
        for user in self.data.user:
            for mp in mPaths:
                if mp == p1:
                    self.walkCount = 10
                if mp == p2:
                    self.walkCount = 8
                if mp == p3:
                    self.walkCount = 8
                if mp == p4:
                    self.walkCount = 5
                if mp == p5:
                    self.walkCount = 5

                for t in range(self.walkCount):
                    path = ['U' + user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'I':
                                    nextNode = choice(self.positive[lastNode])

                                if tp == 'U':
                                    if lastType == 'I':
                                        nextNode = choice(self.pItems[lastNode])
                                    elif lastType == 'F':
                                        nextNode = choice(self.UFNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UTNet[lastNode])

                                if tp == 'F':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
                                        nextNode = choice(self.UFNet[lastNode])

                                if tp == 'T':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
                                        nextNode = choice(self.UFNet[lastNode])

                                path.append(tp + nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.pWalks.append(path)

        self.nWalks = []
        # self.usercovered = {}

        # negative
        for user in self.data.user:
            for mp in mPaths:
                if mp == p1:
                    self.walkCount = 10
                if mp == p2:
                    self.walkCount = 8
                if mp == p3:
                    self.walkCount = 8
                if mp == p4:
                    self.walkCount = 5
                if mp == p5:
                    self.walkCount = 5
                for t in range(self.walkCount):
                    path = ['U' + user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'I':
                                    nextNode = choice(self.negative[lastNode])

                                if tp == 'U':
                                    if lastType == 'I':
                                        nextNode = choice(self.nItems[lastNode])
                                    elif lastType == 'F':
                                        nextNode = choice(self.UFNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UFNet[lastNode])
                                    elif lastType == 'T':
                                        nextNode = choice(self.UTNet[lastNode])
                                        while nextNode not in self.data.user:
                                            nextNode = choice(self.UTNet[lastNode])

                                if tp == 'F':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
                                        nextNode = choice(self.UFNet[lastNode])

                                if tp == 'T':
                                    nextNode = choice(self.UFNet[lastNode])
                                    while nextNode not in self.data.user:
                                        nextNode = choice(self.UFNet[lastNode])

                                path.append(tp + nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.nWalks.append(path)

        shuffle(self.pWalks)
        print('pwalks:', len(self.pWalks))
        print('nwalks:', len(self.nWalks))

    def computeSimilarity(self):
        # Training get top-k friends
        print('Generating user embedding...')
        self.pTopKSim = {}
        self.nTopKSim = {}
        self.pSimilarity = defaultdict(dict)
        self.nSimilarity = defaultdict(dict)
        pos_model = w2v.Word2Vec(self.pWalks, size=50, window=5, min_count=0, iter=10)
        neg_model = w2v.Word2Vec(self.nWalks, size=50, window=5, min_count=0, iter=10)
        for user in self.positive:
            uid = self.data.user[user]
            try:
                self.W[uid] = pos_model.wv['U' + user]
            except KeyError:
                continue
        for user in self.negative:
            uid = self.data.user[user]
            try:
                self.G[uid] = neg_model.wv['U' + user]
            except KeyError:
                continue
        print('User embedding generated.')

        print('Constructing similarity matrix...')
        i = 0
        for user1 in self.positive:
            uSim = []
            i += 1
            if i % 200 == 0:
                print(i, '/', len(self.positive))
            vec1 = self.W[self.data.user[user1]]
            for user2 in self.positive:
                if user1 != user2:
                    vec2 = self.W[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]

            self.pTopKSim[user1] = [item[0] for item in fList]


        i = 0
        for user1 in self.negative:
            uSim = []
            i += 1
            if i % 200 == 0:
                print(i, '/', len(self.negative))
            vec1 = self.G[self.data.user[user1]]
            for user2 in self.negative:
                if user1 != user2:
                    vec2 = self.G[self.data.user[user2]]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2, sim))
            fList = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]
            for pair in fList:
                self.nSimilarity[user1][pair[0]] = pair[1]
            self.nTopKSim[user1] = [item[0] for item in fList]

        self.seededFriends = defaultdict(list)
        self.firend_item_set = defaultdict(list)
        for user in self.pTopKSim:
            trueFriends = list(set(self.pTopKSim[user]).intersection(set(self.nTopKSim[user])))
            self.seededFriends[user] = trueFriends+self.pTopKSim[user][:50]

        for user in self.pTopKSim:
            for friend in self.seededFriends[user]:
                self.firend_item_set[user]+=list(self.data.trainSet_u[friend].keys())

    def sampling(self,vec):

        vec = tf.nn.softmax(vec)

        logits = gumbel_softmax(vec, 0.2)
        return logits


    def build_graph(self):

        self.u = tf.placeholder(tf.int32, name="user_holder")
        self.u_i_matrix = tf.placeholder(tf.float32, name="feedback_matrix")
        self.pos = tf.placeholder(tf.int32, name="positive_item")
        self.fnd = tf.placeholder(tf.int32, name="friend_item")
        self.neg = tf.placeholder(tf.int32, name="neg_holder")
        self.i = tf.placeholder(tf.int32, name="item_holder")

        with tf.name_scope("generator"):

            #AutoEncoder
            initializer = tf.contrib.layers.xavier_initializer()
            self.X = tf.placeholder(tf.float32, [None, self.num_users])
            #self.sample = tf.placeholder(tf.float32, [None, self.num_users])

            self.weights = {
                'encoder': tf.Variable(initializer([self.num_users, 200])),
                'decoder': tf.Variable(initializer([200, self.num_users])),
            }
            self.biases = {
                'encoder': tf.Variable(initializer([200])),
                'decoder': tf.Variable(initializer([self.num_users])),
            }

            self.g_params = [self.weights, self.biases]


            layer = tf.nn.sigmoid(tf.matmul(self.X, self.weights['encoder']) + self.biases['encoder'])
            self.g_output = tf.nn.sigmoid(tf.matmul(layer, self.weights['decoder']) + self.biases['decoder'])


            self.y_pred = tf.multiply(self.X, self.g_output)
            self.y_pred = tf.maximum(1e-6, self.y_pred)

            cross_entropy = -tf.multiply(self.X, tf.log(self.y_pred)) - tf.multiply((1 - self.X),
                                                                                    tf.log(1 - self.y_pred))
            self.reconstruction = tf.reduce_sum(cross_entropy) + self.regU * (
                    tf.nn.l2_loss(self.weights['encoder']) + tf.nn.l2_loss(self.weights['decoder']) +
                    tf.nn.l2_loss(self.biases['encoder']) + tf.nn.l2_loss(self.biases['decoder']))

            g_pre = tf.train.AdamOptimizer(self.lRate)
            self.g_pretrain = g_pre.minimize(self.reconstruction, var_list=self.g_params)



        with tf.variable_scope('discriminator'):

            self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.k], stddev=0.005),dtype=tf.float32)
            self.item_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.k], stddev=0.005),dtype=tf.float32)
            self.item_selection = tf.get_variable('item_selection',initializer=tf.constant_initializer(0.01),shape=[self.num_users, self.num_items])

            self.d_params = [self.user_embeddings, self.item_embeddings,self.item_selection]

            # placeholder definition
            self.u = tf.placeholder(tf.int32,name="u")

            self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u,name='u_e')
            self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos,name='i_e')
            #self.f_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.fnd,name='f_e')
            self.j_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg,name='j_e')
            #self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i,name='f_e')

            #generate virtual friends by gumbel-softmax
            self.virtualFriends = self.sampling(self.g_output) #one-hot

            #get candidate list (items)
            self.candidateItems = tf.matmul(self.virtualFriends, self.u_i_matrix, transpose_a=False,transpose_b=False)

            self.embedding_selection = tf.nn.embedding_lookup(self.item_selection, self.u,name='e_s')

            self.virtual_items = self.sampling(tf.multiply(self.candidateItems,self.embedding_selection))

            self.v_i_embedding = tf.matmul(self.virtual_items,self.item_embeddings,transpose_a=False,transpose_b=False)


            y_us = tf.reduce_sum(tf.multiply(self.u_embedding,self.i_embedding),1)\
                                 -tf.reduce_sum(tf.multiply(self.u_embedding,self.j_embedding),1)

            self.d_pretrain_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_us)))+self.regU*(tf.nn.l2_loss(self.u_embedding)+
                                                                                       tf.nn.l2_loss(self.j_embedding)+
                                                                                       tf.nn.l2_loss(self.i_embedding))

            y_uf = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) - \
                 tf.reduce_sum(tf.multiply(self.u_embedding, self.v_i_embedding), 1)

            y_fs = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_i_embedding), 1)-\
                 tf.reduce_sum(tf.multiply(self.u_embedding, self.j_embedding), 1)



            self.d_loss = -tf.reduce_sum(tf.log(tf.sigmoid(y_uf)))-tf.reduce_sum(tf.log(tf.sigmoid(y_fs)))+\
                          self.regU*(tf.nn.l2_loss(self.u_embedding)+tf.nn.l2_loss(self.i_embedding)+tf.nn.l2_loss(self.j_embedding))
            #
            self.g_loss = 100*tf.reduce_sum(y_uf) #better performance


            d_pre = tf.train.AdamOptimizer(self.lRate)

            self.d_pretrain = d_pre.minimize(self.d_pretrain_loss, var_list=self.d_params)




            self.d_output = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings),1)

        d_opt = tf.train.AdamOptimizer(self.lRate)
        self.d_update = d_opt.minimize(self.d_loss,var_list=self.d_params)
        g_opt = tf.train.AdamOptimizer(self.lRate)
        self.g_update = g_opt.minimize(self.g_loss,var_list=self.g_params)



    def next_batch_d(self,batch_id):

        if batch_id+self.batch_size<=self.train_size:
            users = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
            items = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
            batch_id+=self.batch_size
        else:
            users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
            items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
            batch_id=self.train_size

        u_idx,i_idx,j_idx = [],[],[]
        item_list = list(self.data.item.keys())
        for i,user in enumerate(users):

            i_idx.append(self.data.item[items[i]])
            u_idx.append(self.data.user[user])

            neg_item = choice(item_list)
            while neg_item in self.data.trainSet_u[user]:
                neg_item = choice(item_list)
            j_idx.append(self.data.item[neg_item])

        return batch_id,u_idx,i_idx,j_idx

    def next_batch_g(self,batch_id):
        userList = list(self.data.user.keys())
        if batch_id+self.batch_size<=self.num_users:
            profiles = np.zeros((self.batch_size, self.num_users))
            for i,user in enumerate(userList[batch_id:self.batch_size+batch_id]):
                ind = [self.data.user[friend] for friend in self.seededFriends[user]]
                profiles[i][ind]=1
                batch_id+=self.batch_size

        else:
            profiles = np.zeros((self.num_users-batch_id, self.num_users))
            for i, user in enumerate(userList[self.num_users-batch_id:self.num_users]):
                ind = [self.data.user[friend] for friend in self.seededFriends[user]]
                profiles[i][ind] = 1
                batch_id=self.num_users

        return batch_id,profiles


    def initModel(self):
        super(SRGAN, self).initModel()

        #collect implicit friends

        self.readNegativeFeedbacks()
        self.randomWalks()
        self.computeSimilarity()
        self.build_graph()

        self.matrix = np.zeros(shape=(self.num_users, self.num_items), dtype=np.float32)
        for item in self.data.trainingData:
            u = self.data.user[item[0]]
            i = self.data.item[item[1]]
            self.matrix[u][i] = 1.0



    def buildModel(self):
        # minimax training
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # pretraining

        #print 'pretraining for discriminator...'
        # self.friend_item_set(self.seed_friends)
        # for i in range(30):
        #     batch_id=0
        #     while batch_id<self.train_size:
        #         batch_id,user_idx, i_idx, j_idx = self.next_batch_d(batch_id)
        #         _, loss = self.sess.run([self.d_pretrain, self.d_pretrain_loss],
        #                                 feed_dict={self.u: user_idx, self.neg: j_idx,
        #                                            self.pos:i_idx})
        #
        #         print 'pretraining:', i + 1, 'batch_id',batch_id,'discriminator loss:', loss
        #
        # self.ranking_performance()

        f = open(self.foldInfo+'SRGAN.txt','w')
        res = []

        print('pretraining for generator...')
        for i in range(30):
            batch_id=0
            while batch_id<self.num_users:
                batch_id,profiles = self.next_batch_g(batch_id)
                _,loss = self.sess.run([self.g_pretrain,self.reconstruction],feed_dict={self.X:profiles})
                print('pretraining:', i + 1, 'batch_id',batch_id,'generator loss:', loss)


        print('Training GAN...')

        for i in range(50):
            batch_id = 0
            while batch_id < self.train_size:
                batch_id, user_idx, i_idx, j_idx = self.next_batch_d(batch_id)

                profiles = np.zeros((len(user_idx),self.num_users))
                for n,u in enumerate(user_idx):
                    u_name = self.data.id2user[u]
                    idx = [self.data.user[friend] for friend in self.seededFriends[u_name]]
                    profiles[n][idx]=1


                #generator
                _,loss = self.sess.run([self.g_update,self.g_loss],feed_dict={self.u: user_idx,self.neg:j_idx,
                                                   self.pos: i_idx,self.X:profiles,self.u_i_matrix:self.matrix})
                #discriminator
                _, loss = self.sess.run([self.d_update, self.d_loss],
                                        feed_dict={self.u: user_idx,self.neg:j_idx,
                                                   self.pos: i_idx,self.X:profiles,self.u_i_matrix:self.matrix})

                print('training:', i + 1, 'batch_id', batch_id, 'discriminator loss:', loss)

            results = self.ranking_performance()
            res+=results

        f.writelines(res)


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.user[u]

            # In our experiments, discriminator performs better than generator
            res = self.sess.run(self.d_output, {self.u:u})
            return res

        else:
            return [self.data.globalMean] * self.num_items



