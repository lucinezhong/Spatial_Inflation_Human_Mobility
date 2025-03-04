import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce
import os
from pathlib import Path

# import and set up pyspark configuratio
'''
from pyspark.sql import SparkSession
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import col, length,monotonically_increasing_id,udf,row_number
from pyspark.sql import types as T
from pyspark.sql.window import Window

from pyspark.serializers import MarshalSerializer
# #set an application name
# #start spark cluster; if already started then get it else create it
sc = SparkSession.builder.appName('data_preprocess').master("local[*]").getOrCreate()
# initialize SQLContext from spark cluster
sqlContext = SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)

'''

def load_data(path_data='../data',date='2020010100', temp='',package_for_df='spark'):
    '''import the raw data.
    parameters
        path_data - relative path of the data relative to this script in 'src', e.g., path_data = '../data'
        package - the package used for loading the csv.gz file, including spark and dd (dask)
    '''
    # load raw data
    os.chdir(path_data)
    #path_datafile = os.path.join(os.getcwd(), '{}/*.csv.gz'.format(date))
    path_datafile = os.path.join(os.getcwd(), '{}/{}'.format(date,temp))  # load all csv.gz files at once
    # select the package used to load data, spark, pd (pandas), or dd (dask)
    if package_for_df == 'spark':
        df=sqlContext.read.option("delimiter", "\t").csv(path_datafile)
        #df = sqlContext.read.csv(path_datafile, header=False)
    else:
        df = pd.read_csv(path_datafile, compression='gzip', delimiter='\t',error_bad_lines=False)
    return df




def rename_col(df, package_for_df='spark'):
    '''rename the columns. The columns in the resultant df from loading data with spark are all of string type.
       note: 'time' and 'time_original' are in unix timestamp format (integer value)
       ## change column type
           # # check out data type of each column
           # df.printSchema()  # all string, column names are '_c0' to '_c9'
           # # show top 5 rows:
               # df.show(5)
    '''

    # rename the columns
    col_names_old = df.columns
    col_names_new = ['time', 'id_str', 'device_type', 'latitude', 'longitude', 'accuracy', 'timezone', 'class',
                     'transform']
    if package_for_df == 'spark':
        for i in range(len(col_names_old)):
            df = df.withColumnRenamed(col_names_old[i], col_names_new[i])
    else:
        df = df.rename(columns=dict(zip(col_names_old, col_names_new)))

    # change column type
    if package_for_df == 'spark':
        schema_new = [IntegerType(), StringType(), IntegerType(), FloatType(), FloatType(),
                      FloatType(), IntegerType(), StringType(), StringType(), IntegerType()]
        for i in range(len(col_names_new)):
            df = df.withColumn(col_names_new[i], df[col_names_new[i]].cast(schema_new[i]))
    else:
        schema_new = [int, str, int, float, float, int, int, str, str, int]
        for i in range(len(col_names_new)):
            col = col_names_new
            df[col] == df[col].astype(schema_new[i],errors = 'ignore')

    return df


def select_col(df,package_for_df='spark'):
    '''select columns: id_str, time, latidude, and longitude, accuracy
    '''

    col_select = ['time', 'latitude', 'longitude', 'id_str']
    if package_for_df=='spark':
        df_select = df.select(*col_select)
    else:
        df_select=pd.DataFrame()
        for col in col_select:
            df_select[col]=df[col]

    return df_select

def remove_error_entry(df, package_for_df='spark'):
    '''remove entries with erreneous value of coordinates and 'id_str'.
       There can be errors in the latitude or longitude. E.g., the min of latitude is -14400
    '''

    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180
    id_str_len_min = 15

    if package_for_df == 'spark':
        df = df.filter((df.latitude <= lat_max) & (df.latitude >= lat_min) &
                       (df.longitude <= lon_max) & (df.longitude >= lon_min))

        df = df.filter(length(col('id_str')) > id_str_len_min)
    else:
        df = df[(df.latitude <= lat_max) & (df.latitude >= lat_min) &
                (df.longitude <= lon_max) & (df.longitude >= lon_min)]

        df = df[df.id_str.str.len() > id_str_len_min]

    return df

def load_data_main(path_data,date,temp):
    package_for_df = 'spark'
    package_for_df = ' '
    df = load_data(path_data, date, temp,package_for_df)
    df = rename_col(df,package_for_df)
    df = select_col(df,package_for_df)
    df = remove_error_entry(df,package_for_df)
    return df