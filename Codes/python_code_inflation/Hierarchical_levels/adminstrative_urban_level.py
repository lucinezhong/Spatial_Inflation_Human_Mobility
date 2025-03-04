import pandas as pd
from area import area
import geopandas as gpd
import numpy as np
import shapefile



def comupte_area_size(file_list,gdf_list):
    ####Computer level size
    output_mat=[]
    level = 0
    for file, gdf in zip(file_list, gdf_list):
        column = []
        fields = file.fields
        count=0
        for shape in file.shapeRecords():
            z = [[i[0], i[1]] for i in shape.shape.points[:]]
            # print(z)
            obj = {'type': 'Polygon', 'coordinates': [z]}
            area_size = area(obj)
            column.append(area_size / 1000000)  ####km^2
            output_mat.append([level,count, gdf['area'].mean(), gdf['radius'].mean()])
            count+=1
        gdf['area'] = np.array(column)
        gdf['radius'] = np.sqrt(np.array(column) / 3.14)
        level += 1


    df_adminstrative_area=pd.DataFrame(np.mat(output_mat),columns=['level','area','radius'])


#####build_hierhichal_levels
from shapely.geometry import Polygon, Point, MultiPolygon


def build_hierhichal_levels(df_parent, df_child, typex):
    if typex == 'Polygon':
        columns_parent = []
        columns_child = []
        for index1, row1 in df_child.iterrows():
            geo1 = row1['geometry']
            parent_index = '-1'
            child_index = '-1'
            for index2, row2 in df_parent.iterrows():
                geo2 = row2['geometry']
                parent_index_true = row2['child_index']

                p3 = geo1.intersection(geo2)

                if p3.area > 0:
                    child_index = parent_index_true + '_' + str(index1)
                    parent_index = parent_index_true
                    print(index1, p3.area, geo2.area, p3.area / geo2.area)
                    print(index1, child_index)
                    break

            columns_parent.append(parent_index)
            columns_child.append(child_index)

    df_child['parent_index'] = columns_parent
    df_child['child_index'] = columns_child
    return df_child


if __name__ == "__main__":
    path = 'Dataset/Shapefiles/'

    gdf1 = gpd.read_file(path + "cb_2018_us_region_500k/cb_2018_us_region_500k.shp")
    gdf2 = gpd.read_file(path + "cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
    gdf3 = gpd.read_file(path + "cb_2018_us_county_500k/cb_2018_us_county_500k.shp")
    gdf4 = gpd.read_file(path + "cb_2023_us_cousub_500k/cb_2023_us_cousub_500k.shp")

    gdf_list = [gdf1, gdf2, gdf3, gdf4]

    file1 = shapefile.Reader(path + "cb_2018_us_region_500k/cb_2018_us_region_500k.shp")
    file2 = shapefile.Reader(path + "cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
    file3 = shapefile.Reader(path + "cb_2018_us_county_500k/cb_2018_us_county_500k.shp")
    file4 = shapefile.Reader(path + "cb_2023_us_cousub_500k/cb_2023_us_cousub_500k.shp")

    file_list = [file1, file2, file3, file4]

    comupte_area_size(file_list, gdf_list)
    '''
    gdf1['parent_index'] = ['0' for i in np.arange(len(gdf1))]
    gdf1['child_index'] = ['0' + '_' + str(i) for i in np.arange(len(gdf1))]
    gdf2 = build_hierhichal_levels(gdf1, gdf2, 'Polygon')
    gdf3 = build_hierhichal_levels(gdf2, gdf3, 'Polygon')
    gdf4 = build_hierhichal_levels(gdf3, gdf4, 'Polygon')

    gdf1.to_file("Dataset/gdf1.shp")
    gdf2.to_file("Dataset/gdf2.shp")
    gdf3.to_file("Dataset/gdf3.shp")
    gdf4.to_file("Dataset/gdf4.shp")
    '''