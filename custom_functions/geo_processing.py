from shapely.geometry import MultiPolygon
import numpy as np
from functools import reduce
import geopandas as gpd
from typing import List, Union, Tuple
from shapely.ops import unary_union
from pathlib import Path
from osgeo import gdal


def select_big_from_MP(WS_geometry):
    """

    Function return only biggest polygon 
    from multipolygon WS
    It's the real WS, and not malfunctioned part of it

    """
    if type(WS_geometry) == MultiPolygon:
        big_area = [polygon_area(lats=polygon.exterior.coords.xy[1],
                                 lons=polygon.exterior.coords.xy[0])
                    for polygon in WS_geometry]
        WS_geometry = WS_geometry[np.argmax(big_area)]
    else:
        WS_geometry = WS_geometry
    return WS_geometry


def polygon_area(lats, lons, radius=6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.

    lats and lons are in degrees.
    """
    from numpy import (arctan2, cos, sin, sqrt,
                       pi, append, diff)

    lats, lons = np.deg2rad(lats), np.deg2rad(lons)
    # Line integral based on Green's Theorem, assumes spherical Earth

    # close polygon
    if lats[0] != lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    # colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats) * sin(lons/2)**2
    colat = 2*arctan2(sqrt(a), sqrt(1-a))

    # azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas = diff(colat)/2
    colat = colat[0:-1]+deltas

    # Perform integral
    integrands = (1-cos(colat)) * daz

    # Integrate
    area = abs(sum(integrands))/(4*pi)

    area = min(area, 1-area)
    if radius is not None:  # return in units of radius
        return area * 4 * pi * radius**2
    else:  # return in ratio of sphere total area
        return area


def find_extent(ws):

    def my_ceil(a, precision=0):
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def my_floor(a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    LONS, LATS = ws.exterior.xy
    max_LAT = np.max(LATS)
    max_LON = np.max(LONS)
    min_LAT = np.min(LATS)
    min_LON = np.min(LONS)

    return (my_floor(min_LON, 2), my_ceil(max_LON, 2),
            my_floor(min_LAT, 2), my_ceil(max_LAT, 2))


def create_GDF(shape):
    """

    create geodataframe with given shape
    as a geometry

    """
    gdf_your_WS = select_big_from_MP(WS_geometry=shape)
    # WS from your data
    gdf_your_WS = gpd.GeoSeries([gdf_your_WS])

    # Create extra gdf to use geopandas functions
    gdf_your_WS = gpd.GeoDataFrame({'geometry': gdf_your_WS})
    gdf_your_WS = gdf_your_WS.set_crs('EPSG:4326')

    return gdf_your_WS


def RotM(alpha):
    """ Rotation Matrix for angle ``alpha`` """
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa],
                     [sa,  ca]])


def getSquareVertices(mm, h, phi):
    """ Calculate the for vertices for square with center ``mm``,
        side length ``h`` and rotation ``phi`` """
    hh0 = np.ones(2)*h  # initial corner
    vv = [np.asarray(mm) + reduce(np.dot, [RotM(phi), RotM(np.pi/2*c), hh0])
          for c in range(4)]  # rotate initial corner four times by 90°
    return np.asarray(vv)


def create_AOI(polygons: List[gpd.GeoDataFrame],
               file_path: Union[Path, str],
               file_name: str) -> Union[Tuple[gpd.GeoDataFrame, str], str]:

    gdf = gpd.GeoDataFrame(index=[0])

    gdf['geometry'] = unary_union(
        [poly.geometry.values[0] for poly in polygons]).convex_hull
    gdf = gdf.set_crs(epsg=4326)

    gdf.to_file(f'{file_path}/{file_name}.shp')
    
    if isinstance(gdf, gpd.GeoDataFrame):
        return (gdf, f'{file_path}/{file_name}.shp')
    else:
        return f'Geometry object not a GeoDataFrame! {type(gdf)}'
    
    
def reproject_and_clip(input_raster,
                       output_raster,
                       projection,
                       shapefile: str = '',
                       resolution: float = 0.):
    if resolution:
        if shapefile:
            options = gdal.WarpOptions(cutlineDSName=shapefile,
                                       cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection,
                                       xRes=resolution,
                                       yRes=resolution)
        else:
            options = gdal.WarpOptions(cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection,
                                       xRes=resolution,
                                       yRes=resolution)
    else:
        if shapefile:
            options = gdal.WarpOptions(cutlineDSName=shapefile,
                                       cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection)
        else:
            options = gdal.WarpOptions(cropToCutline=True,
                                       format='GTIFF',
                                       dstSRS=projection)

    gdal.Warp(srcDSOrSrcDSTab=input_raster,
              destNameOrDestDS=output_raster,
              options=options)

    return output_raster


def create_mosaic(file_path: Union[Path, str],
                  file_name: str,
                  tiles: list) -> str:
    file_target = f'{file_path}/{file_name}.vrt'
    mosaic = gdal.BuildVRT(destName=file_target,
                           srcDSOrSrcDSTab=tiles)
    mosaic.FlushCache()

    return file_target
