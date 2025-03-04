

import pandas as pd
import numpy as np
import time
import warnings
import math
import copy
import random
import community as community_louvain
import h3
import networkx as nx
warnings.filterwarnings("ignore")

import Dataset_load
import glob


def level_compute(df_hierarchical_clustering,df_individual,level_list):

    df_exist = df_hierarchical_clustering[df_hierarchical_clustering['label'].isin(df_individual['label_re7'])]

    level = -1
    area = -1
    level_which = -1
    # print(len(df_exist))
    if len(df_exist) > 0:
        for num, column in enumerate(level_list):
            df_exist['count'] = [1] * len(df_exist)
            df_group = df_exist.groupby([column])['count'].sum().reset_index()
            df_group['count'] = df_group['count'] / df_group['count'].sum()
            check = df_group['count'].values >= 0.8
            # print(column,df_group['count'].values)dd
            if True in check:
                level = num+1
                # print('true',level)
                level_which = df_group[check == True][column].values[0]
                # print('true',level_which)
                area = df_hierarchical_clustering[df_hierarchical_clustering[column] == level_which]['area'].sum()

                # area=df_hierarchical_clustering[df_hierarchical_clustering[column]==level_which][column+'_area'].values[0]

                break

    return level, level_which, area


def module_urban_level_process(df_hierarchical_clustering, df_d_r, path_indi_file,level_list):
    df_hierarchical_clustering['area'] = list(
        map(lambda x: h3.cell_area(x, unit='km^2'), df_hierarchical_clustering['label']))



    dict_level = []
    for columns in level_list:
        dictx = dict(zip(df_hierarchical_clustering['label'], df_hierarchical_clustering[columns]))
        dict_level.append(dictx)

    file_list = glob.glob(path_indi_file + '*.csv')

    count = 0
    output_mat = []
    for file_name in file_list:
        count += 1
        df = pd.read_csv(file_name)
        df['label_re7'] = list(map(lambda loc: h3.geo_to_h3(loc[0], loc[1], resolution=7),zip(df['latitude'], df['longitude'])))

        for idx, df_temp in df.groupby(['id']):
            # print(idx)
            df_temp = df_temp[df_temp['cluster'] != -1]
            for cluster, df_tempx in df_temp.groupby(['cluster']):

                df_d_r_temp = df_d_r[(df_d_r['cluster'] == cluster) & (df_d_r['id'] == idx)]
                if len(df_d_r_temp) > 0:
                    d_home = df_d_r_temp['d_home'].values[0]
                    radius = df_d_r_temp['radius'].values[0]
                    # prob_level=level_compute(df_tempx,level_list,dict_level)

                    level, level_which, area = level_compute(df_hierarchical_clustering,df_tempx,level_list)

                    output_mat.append([d_home, radius, level, level_which, area])
                    #print([d_home, radius, level, level_which, area])
    df_result = pd.DataFrame(np.mat(output_mat), columns=['d_home', 'radius', 'level', 'level_which', 'level_area'])

    return df_result




if __name__ == "__main__":
    ####Load data
    case='cluster'
    level_list = ['level1', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7']

    #case='administrative'
    #level_list = ['level1', 'level2', 'level3', 'level4']

    df_hierarchical_clustering, df_d_r, path_indi_file,path_results=Dataset_load.US_hierichal_load(case)
    #df_hierarchical_clustering, df_d_r, path_indi_file, path_results = Dataset_load.Senegal_hierichal_load()
    #df_hierarchical_clustering, df_d_r, path_indi_file, path_results = Dataset_load.Ivory_hierichal_load()

    ####Process start
    df_result=module_urban_level_process(df_hierarchical_clustering, df_d_r, path_indi_file,level_list)
    df_result.to_csv(path_results+'module_urban_level_'+case+'_save.csv')

