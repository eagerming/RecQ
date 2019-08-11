import re

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
import pandas as pd


import os
import os.path

relative_path_root = '..'
rootdir = 'results/music'
datasets = ['varies_C']
# algorithm = ['MostPopular','BPR','SBPR','TBPR','CUNE_BPR','HERP','IF_BPR']
from collections import defaultdict

results = defaultdict(dict)
paths = []

pat = re.compile('^(.+)@(.+)-measure.*')

pat_time = re.compile('^------------------(.*)------------------.*')
pat_top = re.compile('Top (\d+)')
pat_item = re.compile('(.+):(.+)')

algorithm_list = []

for dataset in datasets:
    files = []
    datapath = os.path.join(relative_path_root, rootdir, dataset)
    for parent, dirnames, filenames in os.walk(datapath):
        for filename in filenames:
            result = pat.match(filename)
            if result is None:
                continue
            algor = result.group(1)
            filepath = os.path.join(datapath, filename)

            rows = []
            row = {}
            with open(filepath, 'r') as f:
                for line in f:
                    rr = pat_top.match(line)
                    if rr is not None:
                        if len(row):
                            rows.append(row)
                        row = {'Top': rr.group(1)}
                    else:
                        item_r = pat_item.match(line)
                        if item_r is not None:
                            try:
                                exec("row['{}']={}".format(item_r.group(1), item_r.group(2)))
                            except:
                                print("row['{}']={} can't be assigned!".format(item_r.group(1), item_r.group(2)))
                                row[item_r.group(1)] = item_r.group(2)
                if len(row):
                    rows.append(row)

            al_result = pd.DataFrame(rows)
            al_re = al_result.groupby('Top').aggregate(max)
            if 'C' in al_re:
                al_re.C = int(al_re.C[al_re.C > 0])

            al_re['algorithm'] = algor
            al_re['Top'] = al_re.index
            al_re['Dataset'] = dataset
            algorithm_list.append(al_re)

dataset_result = pd.concat(algorithm_list)
dataset_result.index.name = 'indexx'
dataset_result.drop(['F1', 'MAP'], axis=1, inplace=True)
dataset_result['Top'] = dataset_result['Top'].astype(int)
# dataset_result = dataset_result[dataset_result['Top'] != 10]
dataset_result.sort_values('Top', inplace=True)

# final_result = dataset_result.set_index(['Dataset','Top','algorithm'])

intermedia = dataset_result.pivot_table(index=['algorithm'], columns=['Dataset', 'Top']).transpose()
intermedia.index.names = ['metrics', 'Dataset', 'Top']
final_result = intermedia.pivot_table(index=['Dataset', 'Top', 'metrics'])

with pd.ExcelWriter('test1.xlsx') as writer:
    final_result.to_excel(writer)

a = 1