
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

def haversine(points_a, points_b, radians=False):
    """
    Calculate the great-circle distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.

    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Using this because it is vectorized (stupid fast).
    """

    def _split_columns(array):
        if array.ndim == 1:
            return array[0], array[1]  # just a single row
        else:
            return array[:, 0], array[:, 1]

    if radians:
        lat1, lon1 = _split_columns(points_a)
        lat2, lon2 = _split_columns(points_b)

    else:
        # convert all latitudes/longitudes from decimal degrees to radians
        lat1, lon1 = _split_columns(np.radians(points_a))
        lat2, lon2 = _split_columns(np.radians(points_b))

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
    h = 2 * 6371 * np.arcsin(np.sqrt(d))
    return h  # in kilometers


def interative_hierarchical_clustering(partent_id, df, resolution):
    df['from_label_res'] = list(map(lambda loc: h3.geo_to_h3(loc[0], loc[1], resolution=resolution), df['from_loc']))
    df['to_label_res'] = list(map(lambda loc: h3.geo_to_h3(loc[0], loc[1], resolution=resolution), df['to_loc']))

    df_group = df.groupby(['from_label_res', 'to_label_res'])['trips'].sum().reset_index()
    df_group = df_group[df_group['from_label_res'] != df_group['to_label_res']]
    df_group['log_trips'] = np.log10(df_group['trips'] + 1)

    net_flow = nx.from_pandas_edgelist(df_group, 'from_label_res', 'to_label_res', ['log_trips'])
    node_partitions_dict = community_louvain.best_partition(net_flow, weight='log_trips')

    childen_partition = dict()
    for key, value in node_partitions_dict.items():
        childen_id = partent_id + '-' + str(value)
        if childen_id not in childen_partition.keys():
            childen_partition[childen_id] = []
        childen_partition[childen_id].append(key)

    children_df = dict()
    for key, value in childen_partition.items():
        df_temp = df[(df['from_label_res'].isin(value)) & (df['to_label_res'].isin(value))]
        children_df[key] = df_temp
    return childen_partition, children_df


def add_strx(strx, i):
    stry = ''
    for x in np.arange(i + 1):
        stry = stry + strx[x] + '-'
    return stry[0:-1]


def process(df_flow,path_result):
    dict_parent_df = {'0': df_flow}

    for resolution in np.arange(1, 8):
        print('resolution', resolution)
        dict_update = dict()
        dict_df_update = dict()

        for partient_id, parent_df in dict_parent_df.items():
            # print('parent_id',partient_id)

            children_dict, children_df = interative_hierarchical_clustering(partient_id, parent_df, resolution)

            dict_update.update(children_dict)
            dict_df_update.update(children_df)

        dict_parent_df = dict_df_update

    output_mat = []
    for key, value in dict_update.items():
        strx = key.split('-')
        strx = list([add_strx(strx, i) for i in np.arange(len(strx))])
        for label in value:
            output_mat.append(list([label]) + strx)

    df_hierarchical_clustering = pd.DataFrame(np.mat(output_mat),
                                              columns=['label', 'level8','level7', 'level6', 'level5', 'level4', 'level3',
                                                       'level2', 'level1'])

    df_hierarchical_clustering['loc'] = list(
        map(lambda hex_id: h3.h3_to_geo(hex_id), df_hierarchical_clustering['label']))

    df_hierarchical_clustering['loc_lat'] = list(map(lambda x: x[0], df_hierarchical_clustering['loc']))
    df_hierarchical_clustering['loc_lon'] = list(map(lambda x: x[1], df_hierarchical_clustering['loc']))
    df_hierarchical_clustering.to_csv(path_result+'df_hierarchical_clustering_save.csv')


if __name__ == "__main__":
    ####Load data
    df_flow, path_result = Dataset_load.US_flow_load()
    #df_flow, path_result = Dataset_load.Senegal_flow_load()
    #df_flow, path_result = Dataset_load.Ivory_flow_load()

    ####Process
    process(df_flow, path_result)