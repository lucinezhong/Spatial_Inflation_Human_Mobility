import sys
import glob
import h3
import networkx as nx
import numpy as np
import pandas as pd
import math
import multiprocessing
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from operator import itemgetter
from itertools import groupby
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import community as community_louvain
from pyproj import Transformer
import warnings
warnings.filterwarnings("ignore")

resolution=12

def cells_to_h3(pos):
    new_label = h3.geo_to_h3(pos[0], pos[1], resolution=resolution)
    return new_label


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



def LatLon_To_XY(Lon,Lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)
    xx, yy = transformer.transform(Lon, Lat)
    return xx, yy

def trjactory_label_reset(df):
    df['label'] = list(map(lambda x: cells_to_h3(x), zip(df['latitude'], df['longitude'])))
    df['latitude'] = list(map(lambda x: h3.h3_to_geo(x)[0], df['label']))
    df['longitude'] = list(map(lambda x: h3.h3_to_geo(x)[1], df['label']))
    return df

class extract():
    def __init__(self,df,df_home,vis,path_indi_file,network_weight):
        self.df = self.set_columns_type_data(df)
        self.df_home = self.set_columns_type_home(df_home)
        self.vis = vis ###True, save individual data for visulization
        self.path_indi_file=path_indi_file
        self.network_weight=network_weight

    def set_columns_type_home(self,df_home):
        df_home['id'] = df_home['id'].astype(int)
        df_home = df_home[~df_home['home_lon'].isna()]
        return df_home

    def set_columns_type_data(self,df):
        df['id'] = df['id'].astype(int)
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        return df

    def Trajecotry_Community_Detection_series(self,clustering_method):
        '''
        input: individual trajectory
        :return:  individual modules
        '''
        df = self.df[['id', 'label', 'latitude', 'longitude', 'start', 'end']]

        df_home=self.df_home[['id', 'home_lat', 'home_lon']]
        df = df.merge(df_home, on='id',how='left')
        df = df.drop_duplicates()

        df_community_list = []

        tasks = list([(id_index, df_temp) for id_index, df_temp in df.groupby('id')])

        for id_index, df_temp in tasks:
            try:
                print('id_index',id_index)
                if clustering_method=='Louvain' or clustering_method=='Naive' or clustering_method=='Label':
                    df_community_temp = self.modular_detection_process(id_index, df_temp, self.path_indi_file,clustering_method)
                else:
                    df_community_temp = self.classical_clustering_method(id_index, df_temp, self.path_indi_file,clustering_method)
                df_community_list.append(df_community_temp)
            except:
                pass

        print(len(df_community_list))

        df_community = pd.concat(df_community_list)
        return df_community


    def classical_clustering_method(self,id_index,df_temp,path_indi_file,clustering_method):
        df_temp = df_temp.reset_index()
        home_lat = df_temp.iloc[0]['home_lat']
        home_lon = df_temp.iloc[0]['home_lon']
        df_temp = df_temp.sort_values(by=['start'])
        df_temp['d_home'] = [haversine((i, j), (home_lat, home_lon)) for i, j in zip(df_temp['latitude'], df_temp['longitude'])]

        df_temp=df_temp.dropna(subset=['latitude','longitude'])
        df_temp = df_temp.drop_duplicates(subset=['latitude', 'longitude'])

        X_arry, Y_arry = LatLon_To_XY(df_temp['longitude'], df_temp['latitude'])
        X = np.transpose(np.mat([X_arry, Y_arry]))

        if clustering_method=='Kmeans':
            n_clusters = 5
            ####Clustering
            model = KMeans(n_clusters=n_clusters, max_iter=100, n_init=1).fit(X)
            #print(len(df_temp),len(model.labels_))
            df_temp['temp_cluster'] =list(model.labels_)
            #df_temp['centroids'] = list(model.cluster_centers_)
            print('K-means', np.unique(df_temp['temp_cluster']))
            #print(df_temp[['d_home','temp_cluster']])
        if clustering_method=='Hierarchical':
            n_clusters = 5
            try:
                clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
                clustering_model.fit(X)
                df_temp['temp_cluster'] =list(clustering_model.labels_)
                print('hierarchical', np.unique(df_temp['temp_cluster']))
            except:
                df_temp['temp_cluster']=np.arange(len(df_temp))

        if clustering_method=='DBSCAN':
            ##### meters
            r=1*1000
            min_samples=3

            print(X.shape)

            Cluster = DBSCAN(eps=r, min_samples=min_samples).fit(X)
            df_temp['temp_cluster'] = list(Cluster.labels_)
            df_temp=df_temp[df_temp['temp_cluster']!=-1]
            print('DBSCAN',np.unique(df_temp['temp_cluster']))
        #print(df_temp[['d_home','temp_cluster']])


        output_mat=[]
        df_vis=[]
        for cluster_index,df_here_temp in df_temp.groupby('temp_cluster'):
            center, radius, area, d_home, stay_t, frequency, record_len, counts_near_home = self.sub_cluster_radius_center(
                df_here_temp, home_lat, home_lon)
            #print('result',d_home,radius,area)
            if len(df_here_temp)>3:
                output_mat.append([id_index, cluster_index, len(df_here_temp),record_len,len(df_here_temp), radius, area,stay_t,frequency, d_home, center,counts_near_home])
                df_here_temp['cluster'] = [cluster_index] * len(df_here_temp)
                df_here_temp['dis_to_cluster']= [haversine(center, (i, j)) for i, j in list(zip(df_here_temp['latitude'], df_here_temp['longitude']))]
            else:
                df_here_temp['cluster'] = [-1] * len(df_here_temp)
                df_here_temp['dis_to_cluster'] = [haversine(center, (i, j)) for i, j in list(zip(df_here_temp['latitude'], df_here_temp['longitude']))]
            df_vis.append(df_here_temp)
        if len(output_mat)>0:
            df_cluster_temp = pd.DataFrame(np.array(output_mat),
                                             columns=['id', 'cluster', '#move', '#stay_move', '#unique_loc','radius', 'area', 'stay_t',
                                                      'frequency', 'd_home', 'center','counts_near_home'])
        else:
            df_cluster_temp=pd.DataFrame()
        if self.vis == True and len(df_vis)>0:
            print(len(df_vis),len(df_temp))
            df_vis = pd.concat(df_vis)
            df_vis.to_csv(path_indi_file+ str(id_index) + '.csv')

        return df_cluster_temp


    def modular_detection_process(self,id_index,df_temp,path_indi_file,clustering_method):
        df_temp=df_temp.reset_index()

        df_temp = trjactory_label_reset(df_temp)


        #print('id',id_index)
        home_lat = df_temp.iloc[0]['home_lat']
        home_lon = df_temp.iloc[0]['home_lon']
        df_temp['d_home'] = [haversine((i, j), (home_lat, home_lon)) for i, j in zip(df_temp['latitude'], df_temp['longitude'])]
        df_temp = df_temp[(df_temp['d_home'] > 0) & (df_temp['d_home'] < 4000)]

        data_shift = df_temp.shift(periods=-1)
        df_temp['consecutive_dis'] = [haversine((i, j), (k, l)) for i, j, k, l in zip(df_temp['latitude'], df_temp['longitude'], data_shift['latitude'], data_shift['longitude'])]
        df_temp=df_temp.iloc[0:-1,:]
        df_temp['to_label'] = data_shift['label']

        max_dis=4000

        df_vis=[]
        output_mat = []
        if len(df_temp) > 2 and df_temp['consecutive_dis'].max() > 0:
            if self.network_weight=='log_weight':
                df_temp['weight']=[math.log10(max_dis/(i+1)) for i in df_temp['consecutive_dis']]
            if self.network_weight == 'no_log_weight_reciprocal':
                df_temp['weight'] = [max_dis / (i + 1) for i in df_temp['consecutive_dis']]
                #df_temp['weight']=[df_temp['consecutive_dis'].max()/(i+1) for i in df_temp['consecutive_dis']]
            if self.network_weight =='no_log_weight':
                df_temp['weight'] = [df_temp['consecutive_dis'].max()-i for i in df_temp['consecutive_dis']]

            G = nx.from_pandas_edgelist(df_temp, 'label', 'to_label', ['weight'])

            if clustering_method == 'Louvain':
                partition = community_louvain.best_partition(G, weight='weight')
                communities_generator = dict()
                for value in list(partition.values()):
                    communities_generator[value] = []
                for key, value in partition.items():
                    communities_generator[value].append(key)  ######communities---nodes

            if clustering_method == 'Naive':
                print('true')
                H = G.to_undirected()
                partition=nx.community.greedy_modularity_communities(H, weight='weight')
                partition=list(partition)
                communities_generator = dict()
                for key in np.arange(len(partition)):
                    communities_generator[key]=list(partition[key])
                print(communities_generator)
                #coms = algorithms.leiden(G)

            if clustering_method == 'Label':
                #print('true')
                H = G.to_undirected()
                partition=nx.community.label_propagation_communities(H) #weight='weight'
                partition=list(partition)
                communities_generator = dict()
                for key in np.arange(len(partition)):
                    communities_generator[key]=list(partition[key])
                #print(communities_generator)
                #

            for cluster_index,commu in communities_generator.items():
                #print(cluster_index)
                df_here_temp = df_temp[df_temp['label'].isin(commu)]
                center, radius, area, d_home, stay_t, frequency, record_len, counts_near_home = self.sub_cluster_radius_center(
                    df_here_temp, home_lat, home_lon)
                #print('result',d_home,radius,area)
                if len(np.unique(commu))>2:
                    output_mat.append([id_index, cluster_index, len(df_here_temp),record_len,len(commu), radius, area,stay_t,frequency, d_home, center, counts_near_home])
                    df_here_temp['cluster'] = [cluster_index] * len(df_here_temp)
                    df_here_temp['dis_to_cluster']= [haversine(center, (i, j)) for i, j in list(zip(df_here_temp['latitude'], df_here_temp['longitude']))]
                else:
                    df_here_temp['cluster'] = [-1] * len(df_here_temp)
                    df_here_temp['dis_to_cluster'] = [haversine(center, (i, j)) for i, j in list(zip(df_here_temp['latitude'], df_here_temp['longitude']))]
                df_vis.append(df_here_temp)
        if len(output_mat)>0:
            df_cluster_temp = pd.DataFrame(np.array(output_mat),
                                             columns=['id', 'cluster', '#move', '#stay_move', '#unique_loc','radius', 'area', 'stay_t',
                                                      'frequency', 'd_home', 'center', 'counts_near_home'])
        else:
            df_cluster_temp=pd.DataFrame()
        if self.vis == True and len(df_vis)>0:
            print(len(df_vis),len(df_temp))
            df_vis = pd.concat(df_vis)
            df_vis.to_csv(path_indi_file+ str(id_index) + '.csv')

        return df_cluster_temp

    def sub_cluster_stay(self,df_temp):
        index_list=df_temp.index
        stay_t=[]
        record_len=[]
        frequency=0
        #print('intial_length',len(df_temp))
        for k, g in groupby(enumerate(index_list), lambda ix : ix[0] - ix[1]):
            group=list(map(itemgetter(1), g))

            df_temp_temp=df_temp[df_temp.index.isin(group)]
            #print('group',group,'len',len(df_temp_temp))
            stay_t.append((df_temp_temp['end'].max() - df_temp_temp['start'].min()) / 3600)
            record_len.append(len(df_temp_temp))
            frequency+=1
        stay_t = np.mean(stay_t)
        record_len=np.mean(record_len)
        return stay_t,frequency,record_len


    def sub_cluster_radius_center(self, data,home_lat,home_lon):
        ####for check no near_home_lcoations
        points = list(zip(data['latitude'], data['longitude']))
        center = (data['latitude'].mean(), data['longitude'].mean())
        d_home = haversine(center, (home_lat, home_lon))

        radius_list = [haversine(center, (i, j)) for i, j in points]
        dis_dict=dict(zip(points,radius_list))
        dis_dict=dict(sorted(dis_dict.items(), key=lambda item: item[1]))

        radius = np.mean(list(dis_dict.values())[1:-1])
        points=list(zip([i[0] for i in list(dis_dict.keys())][1:-1],[i[1] for i in list(dis_dict.keys())][1:-1]))
        center = (np.mean([i[0] for i in list(dis_dict.keys())][1:-1]), np.mean([i[1] for i in list(dis_dict.keys())][1:-1]))

        try:
            area=ConvexHull(points).area
        except:
            area=0
        d_home = haversine(center, (home_lat, home_lon))
        #print('d_home',d_home)
        stay_t,frequency ,record_len= self.sub_cluster_stay(data) ####in hour
        counts_near_home=len(data[data['d_home']<=1])

        return center, radius,area, d_home,stay_t,frequency,record_len,counts_near_home











