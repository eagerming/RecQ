import re

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
import pandas as pd

import os
import os.path
from collections import defaultdict
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter




def plot_double_column(table1, title1, xlabel1, ylabel1,
                       table2, title2, xlabel2, ylabel2,
                       table3, title3, xlabel3, ylabel3,
                       savepath, show=True, is_equal_y=True):
    df1 = table1
    df2 = table2
    df3 = table3
    legend1 = list(table1.columns)

    dash_styles = ["", (1, 1), (4, 1.5),  (3, 1, 1.5, 1), (5, 1, 1, 1), (5, 1, 2, 1, 2, 1), (2, 2, 3, 1.5),
                   (1, 2.5, 3, 1.2), (3, 1), (1, 2, 3, 4),
                   "", (4, 1.5), (1, 1), (3, 1, 1.5, 1), (5, 1, 1, 1), (5, 1, 2, 1, 2, 1), (2, 2, 3, 1.5),
                   (1, 2.5, 3, 1.2), (3, 1), (1, 2, 3, 4)]
    filled_markers = ('o',  'X', '*', 'h', 's', 'v', '^', '<', '>', 'p', '8', 'H', 'D', 'd', 'P',
                      'o',  'X', '*', 'h', 's', 'v', '^', '<', '>', 'p', '8', 'H', 'D', 'd', 'P',)

    import matplotlib

    SMALL = 7.5
    Regular = 9.5
    matplotlib.rc('font', size=Regular)
    matplotlib.rc('axes', titlesize=Regular)
    # matplotlib.rcParams['axes.linewidth'] = .1

    # sns.set_style("whitegrid")
    # sns.set(style="ticks", rc={"lines.linewidth": 0.7})
    # sns.set(rc={"lines.linewidth": 0.1})

    fig = plt.figure(figsize=(5, 2.5))
    # fig, ax = plt.subplots(1,2)
    # ax1 = fig.add_subplot(1,2,1)
    ax1 = fig.add_axes([0.12, 0.15, 0.23, 0.6])
    # with plt.rc_context({'lines.linewidth': 0.3}):
    #     sns.lineplot(data=df1, palette="Set2", ax=ax1, linewidth=2, dashes=dash_styles, markers=filled_markers)
    sns.lineplot(data=df1, palette="Set2", ax=ax1, linewidth=1.5, dashes=dash_styles,markers=filled_markers, markeredgewidth=0, legend=False)
    # plt.yticks(rotation=90)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax2 = fig.add_axes([0.44, 0.15, 0.23, 0.6])
    sns.lineplot(data=df2, palette="Set2", ax=ax2, linewidth=1.5, dashes=dash_styles,markers=filled_markers, markeredgewidth=0,  legend=False)
    ax3 = fig.add_axes([0.76, 0.15, 0.23, 0.6])
    sns.lineplot(data=df3, palette="Set2", ax=ax3, linewidth=1.5, dashes=dash_styles,markers=filled_markers, markeredgewidth=0,  legend=False)

    ax1.legend(labels=legend1, ncol=4, loc='lower left', bbox_to_anchor=(.6, 1.1))
    # ax1.set_title(title1)
    ax1.set_xlabel(xlabel1)
    ax1.set_ylabel(ylabel1)
    # ax1.set_xticks(list(range(len(index1) + 1)[::5]))
    ax1.set_xticks([2, 4, 6, 8, 10])
    ax1.xaxis.set_label_coords(0.5, -0.15)
    ax1.xaxis.set_tick_params(labelsize=SMALL)
    ax1.yaxis.set_tick_params(labelsize=SMALL)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # ax1.grid(linestyle='-.')

    # ax2.set_title(title2)
    ax2.set_xlabel(xlabel2)
    # ax2.set_ylabel(ylabel2)
    # ax2.set_xticks(list(range(len(index2) + 1)[::2]))
    ax2.set_xticks([2,4,6,8,10])
    # ax2.set_yticks([0.65,0.7,0.75,0.8,0.85])
    ax2.xaxis.set_label_coords(0.5, -0.15)
    ax2.xaxis.set_tick_params(labelsize=SMALL)
    ax2.yaxis.set_tick_params(labelsize=SMALL)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    if is_equal_y:
        ax2.set_ylim(ax1.get_ylim())

    # ax3.set_title(title3)
    ax3.set_xlabel(xlabel3)
    # ax3.set_ylabel(ylabel3)
    # ax2.set_xticks(list(range(len(index2) + 1)[::2]))
    ax3.set_xticks([2, 4, 6, 8, 10])
    # ax2.set_yticks([0.65,0.7,0.75,0.8,0.85])
    ax3.xaxis.set_label_coords(0.5, -0.15)
    ax3.xaxis.set_tick_params(labelsize=SMALL)
    ax3.yaxis.set_tick_params(labelsize=SMALL)
    if is_equal_y:
        ax3.set_ylim(ax1.get_ylim())
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))


    # ax2.grid(linestyle='-.')

    plt.savefig(savepath, format='eps')
    plt.savefig(savepath[:-4] + '.pdf', format='pdf')
    if show:
        plt.show()


def get_table(param, relative_path_root='..', rootdir='results/music', datasets=['varies_N'],
              metric='Recall', top=10):

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
                al_re = al_result.groupby('Top').aggregate(np.mean)
                if 'C' in al_re:
                    al_re.C = int(al_re.C[al_re.C > 0])
                if 'N' in al_re:
                    al_re.N = int(al_re.N[al_re.N >= 0])
                if 'K' in al_re:
                    al_re.K = int(al_re.K[al_re.K >= 0])
                if 'L' in al_re:
                    al_re.L = int(al_re.L[al_re.L >= -1])

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

    table_top = dataset_result[dataset_result.Top == top]
    table_measure = table_top.pivot_table(columns=['algorithm'], index=[param], values=[metric])[metric]

    table_measure = table_measure[['ABPR_sqrt', 'ABPR_t1']]
    table_measure.rename(columns={'ABPR_sqrt': 'SA-BPR', 'ABPR_t1': 'SA-BPR without $\mathcal{I}_o$'}, inplace=True)

    # try:
    #     table_measure = table_measure[['ABPR_sqrt', 'ABPR_t1']]
    #     table_measure.rename(columns={'ABPR_sqrt':'SA-BPR', 'ABPR_t1':'SA-BPR without $\mathcal{I}_2$'}, inplace=True)
    # except:
    #     table_measure = table_measure[['ABPR_sqrt', 'ABPR_d']]
    #     table_measure.rename(columns={'ABPR_sqrt': 'SA-BPR', 'ABPR_d': 'SA-BPR without $\mathcal{I}_2$'}, inplace=True)


    return table_measure


def draw_N_varies_dataset():
    top = 20

    relative_path_root = '..'
    metric = 'Recall'
    param = 'N'
    datasets = ['varies_{}'.format(param)]


    rootdir = 'results/movie'
    table_movie = get_table(param, relative_path_root, rootdir, datasets, metric, top)

    # rootdir = 'results/book'
    # table_book = get_table(param, relative_path_root, rootdir, datasets, metric, top)

    rootdir = 'results/music'
    table_music = get_table(param, relative_path_root, rootdir, datasets, metric, top)



    plot_double_column(table_movie, 'dd', 'Movie', "{}@{}".format(metric, top),
                       table_music, 'dd', 'Book', "{}@{}".format(metric, top),
                       table_music, 'dd', 'Music', "{}@{}".format(metric, top),
                       'measure_param.eps', show=True, is_equal_y=False)


def draw_varies_params():
    top = 20
    rootdir = 'results/music'
    relative_path_root = '..'
    metric = 'Recall'



    param = 'L'
    datasets = ['varies_{}'.format(param)]
    table_L = get_table(param, relative_path_root, rootdir, datasets, metric, top)
    table_L.index = table_L.index + 1

    param = 'N'
    datasets = ['varies_{}'.format(param)]
    table_N = get_table(param, relative_path_root, rootdir, datasets, metric, top)

    # param = 'C'
    # datasets = ['varies_{}'.format(param)]
    # table_C = get_table(param, relative_path_root, rootdir, datasets, metric, top)

    param = 'K'
    datasets = ['varies_{}'.format(param)]
    table_K = get_table(param, relative_path_root, rootdir, datasets, metric, top)



    plot_double_column(table_N, 'dd', 'N', "{}@{}".format(metric, top),
                       table_L, 'dd', 'Length', "{}@{}".format(metric, top),
                       table_K, 'dd', 'K', "{}@{}".format(metric, top),
                       'measure_param.eps', show=True)


if __name__ == '__main__':
    # draw_N_varies_dataset()
    draw_varies_params()



# intermedia = dataset_result.pivot_table(index=['algorithm'], columns=['Dataset', 'Top']).transpose()
# intermedia.index.names = ['metrics', 'Dataset', 'Top']
# final_result = intermedia.pivot_table(index=['Dataset', 'Top', 'metrics'])
#
# with pd.ExcelWriter('test1.xlsx') as writer:
#     final_result.to_excel(writer)
