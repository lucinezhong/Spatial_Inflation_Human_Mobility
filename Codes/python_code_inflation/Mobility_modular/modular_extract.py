
import pandas as pd
import numpy as np
import time
import warnings
import math
import copy
import random
warnings.filterwarnings("ignore")

import modular_extract_fun
import Dataset_load

def extract_process(df,df_home,each_indi_show,path_indi_file,network_weight,cluster_algorithm):
    '''
    :param df: individual trajectory
    :param df_home: individual home
    :param each_indi_show: whether save detailed outcome
    :param path_indi_file: path to save detailed outcome
    :param network_weight: selection of network weight
    :param cluster_algorithm: selection of cluster algorithm
    :return: trajectory cluster/modules
    '''
    start_time = time.time()

    process = modular_extract_fun.extract(df, df_home, each_indi_show, path_indi_file,network_weight)
    cluster_df = process.Trajecotry_Community_Detection_series(cluster_algorithm)

    end_time = time.time()
    print('end_time', end_time - start_time)

    return cluster_df


def downsampling_test(df,df_home,downsampling_ratio,path_results):
    '''
    :param df: individual trajectory
    :param df_home: individual home
    :param downsampling_ratio: downsampling_ratio
    :param path_results: path_results
    :return: sampled trajectory
    '''
    df['id'] = df['id'].astype(int)
    df_home['id'] = df_home['id'].astype(int)
    df_home = df_home[['id', 'home_lat', 'home_lon']]
    df = df.merge(df_home, on='id', how='left')

    frequency_df_list=[]
    new_df_list=[]
    for id, df_temp in df.groupby(['id']):
        df_temp = df_temp.drop_duplicates()
        df_temp = df_temp.dropna()

        df_temp['index']=np.arange(len(df_temp))
        df_temp['d_home'] = [modular_extract_fun.haversine((i, j), (k,l)) for i,j,k,l in
                         zip(df_temp['latitude'], df_temp['longitude'],df_temp['home_lat'], df_temp['home_lon'])]

        df_temp['d_home'] = list(map(lambda x: int(x/10+1)*10, df_temp['d_home']))

        df_frequency=df_temp.groupby(['id','d_home'])['label'].count().reset_index()
        df_frequency['prob']=df_frequency['label']/df_frequency['label'].sum()
        frequency_df_list.append(df_frequency)

        dictx=dict(zip(df_frequency['d_home'],df_frequency['prob']))


        df_temp['new_prob'] = list(map(lambda x: downsampling_ratio if x < 100 else 1, df_temp['d_home']))

        ####do downsampling
        select_index_list=list(map(lambda x: x[0] if random.uniform(0, 1)<x[1] else -1, zip(df_temp['index'],df_temp['new_prob'])))
        df_temp_new=df_temp[df_temp['index'].isin(select_index_list)]

        #print(len(df_temp),len(df_temp_new))
        new_df_list.append(df_temp_new)

    new_df_list=pd.concat(new_df_list)
    frequency_df_list=pd.concat(frequency_df_list)
    frequency_df_list.to_csv(path_results+'d_home_number_record.csv')
    return new_df_list




if __name__ == "__main__":
    ########------------select dataset------------########
    df, df_home, path_results = Dataset_load.sample_dataset_load()
    print(path_results)

    ########------------select network_weight------------########
    network_weight='log_weight'
    #network_weight ='no_log_weight'
    #network_weight = 'no_log_weight_reciprocal'


    ########------------select clustering algorithm------------########
    clustering_algorithm = 'Louvain'
    #clustering_algorithm = 'Naive'
    #clustering_algorithm = 'Label'
    #clustering_algorithm = 'Hierarchical'
    #clustering_algorithm = 'Kmeans'
    #clustering_algorithm = 'DBSCAN'



    ########------------record individual module------------########
    #each_indi_show = False
    each_indi_show = False

    #####------------downsampling_test------------########
    downsampling_test=False
    if downsampling_test==True:
        downsampling_ratio=0.5
        df=downsampling_test(df,df_home,downsampling_ratio,path_results)

    #####------------start extracting modules------------########
    print('start extracting modules')
    community_df=extract_process(df,df_home,each_indi_show,path_results,network_weight,clustering_algorithm)
    community_df.to_csv(path_results + 'results_cluster_'+network_weight+'_'+clustering_algorithm+'.csv')





