import pickle
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import infostop
import sys
sys.stdout.flush()
import multiprocessing as mp
import time
import h3
import math
from datetime import datetime



def finding_home_locations(df_individual):
    '''
    :param df_individual:
    :return: home_label, home_lat, home_lon for individual
    '''
    night_list = [20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    monthly_home = []
    ######home
    df_temp = df_individual[df_individual['start_h'].isin(night_list)]
    if len(df_temp) > 5:
        (home_label) = df_temp.groupby(['label']).size().idxmax()
        home_lat = df_temp[df_temp['label'] == home_label]['latitude'].mean()
        home_lon = df_temp[df_temp['label'] == home_label]['longitude'].mean()
    else:
        home_label = math.nan;
        home_lat = math.nan;
        home_lon = math.nan;

    return [home_label, home_lat, home_lon]


def infer_indiv_stoppoint(individual_list, array_list, path_stopoints, file_name):
    '''
    infer the stop points of each individual given their trajectory data

    Parameters:list of trajectory, list of susers

    Returns: dataframe of stopoints

    '''
    r1, r2 = 30, 30  ####r2=0, no community computed
    min_staying_time, max_time_between = 600, 86400  # in seconds
    model_infostop = infostop.Infostop(r1=r1, r2=r2,
                                       label_singleton=False,
                                       min_staying_time=min_staying_time,
                                       max_time_between=max_time_between,
                                       min_size=2)

    labels_list = model_infostop.fit_predict(array_list)

    output_columns = ['id_str', 'label', 'start', 'end', 'lat_start', 'lon_start', 'lat_end', 'lon_end']
    output_mat = []
    for labels, k in zip(labels_list, np.arange(len(individual_list))):
        position_lat_dict = dict(zip(array_list[k][:, 2], array_list[k][:, 0]))
        position_lon_dict = dict(zip(array_list[k][:, 2], array_list[k][:, 1]))
        trajectory = infostop.postprocess.compute_intervals(labels, array_list[k][:, 2])
        for i in range(len(trajectory)):
            label_index = trajectory[i][0]
            if label_index != -1:
                from_time = trajectory[i][1]
                to_time = trajectory[i][2]
                output_mat.append(
                    [individual_list[k]] + trajectory[i] + [position_lat_dict[from_time], position_lon_dict[from_time]]
                    + [position_lat_dict[to_time], position_lon_dict[to_time]])
                # print(output_mat[-1])
    if len(output_mat) != 0:
        output_mat = np.array(output_mat)
        df_output = pd.DataFrame(data=output_mat, columns=output_columns)
        df_output.to_csv(path_stopoints + "/" + file_name + '.csv')


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