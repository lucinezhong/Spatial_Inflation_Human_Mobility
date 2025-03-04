import pandas as pd
import h3

#####Sample dataset load
def sample_dataset_load():
    path='/Users/luzhong/Documents/GitHub/Spatial_Inflation_Human_Mobility/Codes/'
    path_results='/Users/luzhong/Documents/GitHub/Spatial_Inflation_Human_Mobility/Codes/sample_reuslts/'
    df=pd.read_csv(path+'sample_individual_trajectory.csv')
    df_home=pd.read_csv(path+'sample_individual_home.csv')
    return df, df_home, path_results