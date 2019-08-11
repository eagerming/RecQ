# from Main.load_assignment_data import load_account_assignment_data, load_certain_new_user_item, load_new_table
import os
import pickle
from collections import defaultdict
from itertools import permutations
import pandas as pd
import numpy as np
from tqdm import tqdm


class AccountDAO(object):
    def __init__(self, conf):
        self.config = conf

        account_data_path = conf['account.data']
        identification_algorithm = conf['account.algorithm']
        pickle_data_path = os.path.join(account_data_path, 'recommendation_data_{}.pickle'.format(identification_algorithm))
        if os.path.exists(pickle_data_path):
            print('pickle load data')
            dataset = pickle.load(open(pickle_data_path, 'rb'))
            self.training_user_item, self.training_account_item, self.relation, self.test_user_item, self.test_table, self.ground_visit, self.map_from_user_to_account, self.SI, self.SB, self.reserved_item_set = dataset
        else:
            print('dataset is loading from scratch...')
            self.training_user_item, self.training_account_item, self.relation, self.SI, self.SB, self.reserved_item_set = load_account_assignment_data(conf)
            self.test_table = load_new_table(conf, self.reserved_item_set)
            self.test_user_item = load_certain_new_user_item(self.test_table)
            self.ground_visit, self.map_from_user_to_account = load_ground_truth_history(conf)
            dataset = (self.training_user_item, self.training_account_item, self.relation, self.test_user_item, self.test_table, self.ground_visit, self.map_from_user_to_account, self.SI, self.SB, self.reserved_item_set)
            print('dataset is loaded, pickle dumping...')
            pickle.dump(dataset, open(pickle_data_path, 'wb'))
            print('pickle dump done!')

def load_new_table(conf, reserved_item_set):
    account_data_path = conf['account.data']
    identification_algorithm = conf['account.algorithm']

    test_data_path = os.path.join(account_data_path, 'new_identification',
                                  'new_identification_{}.csv'.format(identification_algorithm))

    print('reading and mapping test table...')
    test_table = pd.read_csv(test_data_path)
    test_table = test_table.loc[:, ~test_table.columns.str.contains('^Unnamed')]
    test_table.account_contain = test_table.account_contain.map(eval)
    test_table = test_table[np.array([len(i) for i in test_table.account_contain]) > 1]

    test_table.item_id_list = transfer_string_seperating_by_space_to_list(test_table.item_id_list)
    test_table.identify_user = transfer_string_seperating_by_space_to_list(test_table.identify_user)


    reduced_item_list = []
    reduced_identify_user_list = []

    for items, identify_users in tqdm(zip(test_table.item_id_list, test_table.identify_user), desc='remove inactive item in test data', total=len(test_table)):
        new_items = []
        new_users = []
        for item, user in zip(items, identify_users):
            if item in reserved_item_set:
                new_items.append(item)
                new_users.append(user)
        reduced_item_list.append(new_items)
        reduced_identify_user_list.append(new_users)

    test_table.item_id_list = reduced_item_list
    test_table.identify_user = reduced_identify_user_list

    return test_table

def load_assignment_table(conf):
    account_data_path = conf['account.data']
    identification_algorithm = conf['account.algorithm']

    training_data_path = os.path.join(account_data_path, 'assignment',
                                      'assignment_{}.csv'.format(identification_algorithm))

    train_table = pd.read_csv(training_data_path)
    train_table = train_table.loc[:, ~train_table.columns.str.contains('^Unnamed')]
    train_table.item_id_list = train_table.item_id_list.map(eval)

    return train_table


def get_reserved_items(train_table, threshold=10):
    reserved_item_set = []
    item_frequency = defaultdict(int)

    for items in train_table.item_id_list:
        for item in set(items):
            item_frequency[item] += 1

    for k, v in item_frequency.items():
        if v >= threshold:
            reserved_item_set.append(k)
    print('reserved items number {}/{}={}'.format(len(reserved_item_set), len(item_frequency),
                                                  len(reserved_item_set)/ len(item_frequency)))
    return reserved_item_set


def load_account_assignment_data(conf):
    train_table = load_assignment_table(conf)
    reserved_item_set = get_reserved_items(train_table)

    training_user_item = []
    training_account_item = []
    relation = []
    SI = {}
    SB = {}

    for ind, row in tqdm(train_table.iterrows(), desc='loading training data...', total=len(train_table)):
        user = row.user_id
        account = row.account_id
        SI[user] = row.SI
        SB[user] = row.SB

        for item in row.item_id_list:
            if item in reserved_item_set:
                training_user_item.append([user, item, 1])
                training_account_item.append([account, item, 1])

    print('handling training table...')
    account_user_map = train_table.groupby('account_id')['user_id'].aggregate(list)
    for users in account_user_map:
        pairs = list(permutations(users,2))
        relation += pairs

    relations = list(map(lambda pair: [*pair,1], relation))
    return training_user_item, training_account_item, relations, SI, SB, reserved_item_set



def transfer_string_seperating_by_space_to_list(series):
    return series.map(lambda a:eval('[{}]'.format(','.join(a.strip()[1:-1].split()))))


def load_test_account_and_user_info(conf):
    pass

def load_certain_new_user_item(test_table):
    # if test_table is None:
    #     test_table = load_new_table(conf)

    k = np.unique(test_table.k)[0]
    table = test_table[test_table['k'] == k]

    # len_table = [len(items) for items in table['item_id_list']]
    # has_len = [len_item >= length_threshold for len_item in len_table]

    test_user_item = []
    for ind, row in table[['truth_user', 'item_id_list']].iterrows():
        for item in row['item_id_list']:
            test_user_item.append([row['truth_user'],item,1])
    return test_user_item

def load_ground_truth_history(conf):
    account_data_path = conf['account.data']
    groundtruth_path = os.path.join(account_data_path, 'ground_truth_table.csv')

    ground_table = pd.read_csv(groundtruth_path)
    ground_table = ground_table.loc[:, ~ground_table.columns.str.contains('^Unnamed')]

    account_user = ground_table.groupby('account_id')['user_id'].aggregate(list)

    map_from_user_to_account = {u: a for a, users in account_user.to_dict().items() for u in users}

    ground_visit = defaultdict(dict)
    for ind, row in tqdm(ground_table[['user_id', 'item_id_list']].iterrows(), desc='loading ground truth', total=len(ground_table)):
        ground_user = row['user_id']
        items = eval(row['item_id_list'])

        ground_visit[ground_user].update({item:1 for item in items})

    return ground_visit, map_from_user_to_account



