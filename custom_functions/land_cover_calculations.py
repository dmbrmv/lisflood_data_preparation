from custom_functions.nc_processing import select_NC_by_extent
from custom_functions.geo_processing import (getSquareVertices, polygon_area,
                                             find_extent)
import xarray as xr
import numpy as np


import itertools
import pandas as pd
from math import dist

from rasterio.mask import mask
from shapely.geometry import mapping, Polygon
import rioxarray
import rasterio
from osgeo import gdal, gdalconst


def calc_spam_part(spam_ds: pd.DataFrame,
                   target_ws: Polygon,
                   crop_variable: str,
                   polygons: list,
                   nc_file_to_save: str,
                   tif_file_to_save: str,
                   grid: xr.Dataset,
                   grid_res: str,
                   lats_grid: np.ndarray,
                   lons_grid: np.ndarray):
    min1_min5 = 1
    min3_min5 = 1
    square_5min = Polygon(getSquareVertices(
        mm=(target_ws.centroid.xy[0][0],
            target_ws.centroid.xy[1][0]),
        h=5/60/2,
        phi=0))
    square_5min_area = polygon_area(lats=square_5min.exterior.coords.xy[1],
                                    lons=square_5min.exterior.coords.xy[0]) / 10**6
    if grid_res == '1':
        square_1min = Polygon(getSquareVertices(
            mm=(target_ws.centroid.xy[0][0],
                target_ws.centroid.xy[1][0]),
            h=1/60/2,
            phi=0))
        square_1min_area = polygon_area(lats=square_1min.exterior.coords.xy[1],
                                        lons=square_1min.exterior.coords.xy[0]) / 10**6
        min1_min5 = square_1min_area/square_5min_area
    if grid_res == '3':
        square_3min = Polygon(getSquareVertices(
            mm=(target_ws.centroid.xy[0][0],
                target_ws.centroid.xy[1][0]),
            h=3/60/2,
            phi=0))
        square_3min_area = polygon_area(lats=square_3min.exterior.coords.xy[1],
                                        lons=square_3min.exterior.coords.xy[0]) / 10**6
        min3_min5 = square_3min_area/square_5min_area

    # define extent from ws
    min_lon, max_lon, min_lat, max_lat = find_extent(target_ws)

    # because of different grids add .15 degrees to cover smaller watersheds
    # select values from SPAM with tangible values
    lons_mask = ((min_lon - .15) <=
                 spam_ds.x) & (spam_ds.x <= (max_lon + .15))
    lats_mask = ((min_lat - .15) <=
                 spam_ds.y) & (spam_ds.y <= (max_lat + .15))
    lat_lon_mask = lats_mask & lons_mask
    spam_ds = spam_ds[lat_lon_mask]
    # get coords bounds from selected dataframe
    pseudo_x_min, pseudo_y_min, pseudo_x_max, pseudo_y_max = spam_ds[
        ['x', 'y']].describe()[['x', 'y']].loc[['min', 'max'], :].values.ravel()

    # supplementary function to compare x, y coordinates of
    # computational grid and SPAM dataset
    def find_resolution(coord_series):

        vals, counts = np.unique(np.diff(np.unique(coord_series)),
                                 return_counts=True)

        return vals[np.argmax(counts)]

    def compare_x_y_res(x_res, y_res):

        if round(x_res, 2) == round(y_res, 2):
            return x_res
        else:
            raise Exception(f'Grid are not equal! x -- {x_res}, y -- {y_res}')

    # create pseudo grid based on resolution
    resolution = compare_x_y_res(x_res=find_resolution(spam_ds['x']),
                                 y_res=find_resolution(spam_ds['y']))
    pseudo_lons = np.arange(start=pseudo_x_min,
                            stop=pseudo_x_max,
                            step=resolution)
    pseudo_lats = np.arange(start=pseudo_y_min,
                            stop=pseudo_y_max,
                            step=resolution)

    pseudo_pairs = list(itertools.product(pseudo_lons, pseudo_lats))

    spam_ds['coord_pairs'] = list(zip(spam_ds['x'],
                                      spam_ds['y']))

    values = dict()

    for pseudo_pair in pseudo_pairs:

        distance_array = [round(dist(pseudo_pair, coord), 1)
                          for coord in spam_ds['coord_pairs']]
        if np.min(distance_array) == 0.0:
            values[pseudo_pair] = (spam_ds[crop_variable].iloc[
                np.argmin(distance_array)]) / square_5min_area * 0.01
        else:
            values[pseudo_pair] = np.NaN

    vals = np.array(list(values.values())).reshape((pseudo_lons.shape[0],
                                                    pseudo_lats.shape[0]))

    test_ds = xr.Dataset(data_vars={f'{crop_variable}': (['lat', 'lon'],
                                                         vals)},
                         coords=dict(lat=(['lat'], pseudo_lats),
                                     lon=(['lon'], pseudo_lons),
                                     ),
                         attrs=dict(
        description=f'{crop_variable}'))

    test_ds.rio.write_crs(4326, inplace=True).rio.set_spatial_dims(
        x_dim="lon",
        y_dim="lat",
        inplace=True).rio.write_coordinate_system(inplace=True)
    # from rasterio import Affine
    test_ds.to_netcdf(f'{nc_file_to_save}.nc')

    gdal.Translate(destName=f'{tif_file_to_save}.tif',
                   srcDS=f'{nc_file_to_save}.nc',
                   outputType=gdalconst.GDT_Float32,
                   metadataOptions='VS_SCALAR')

    res = list()
    with rasterio.open(f'{tif_file_to_save}.tif') as src:
        for poly in polygons:
            test_shp = [mapping(poly.geometry.values[0])]
            out_img, _ = mask(src, test_shp,
                              nodata=np.NaN,
                              all_touched=True,
                              crop=True,
                              pad=True)
            res.append(out_img[0])

    res_ds = np.array([np.nanmean(res_interim)
                       for res_interim in res]).reshape(
        grid.mask.shape)

    if grid_res == '1':
        res_ds *= min1_min5
    elif grid_res == '3':
        res_ds *= min3_min5

    return res_ds


def weighted_lc_value(nc_dataset: str,
                      polygons: list,
                      shape: tuple):
    ds = xr.open_dataset(nc_dataset, engine='rasterio')

    result = list()

    for poly in polygons:
        selection = select_NC_by_extent(ds,
                                        poly.geometry.values[0]
                                        ).band_data.values
        selection = np.mean(selection)
        result.append(selection)
    result = np.array(result).reshape(shape)

    return result


def crop_sum(a, b):
    return np.where(np.isnan(a+b), np.where(np.isnan(a),
                                            b, a),
                    a+b)


def final_crop(crop_grids: list,
               grid_shape: tuple):

    fin_arr = np.empty(shape=grid_shape)
    fin_arr[:] = np.NaN

    for crop_arr in crop_grids:
        fin_arr = crop_sum(crop_arr, fin_arr)

    return np.nan_to_num(fin_arr)
