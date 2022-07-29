from typing import Tuple
import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import Polygon
from .geo_processing import (find_extent, getSquareVertices,
                             create_GDF, select_big_from_MP)


def lisflood_initial_grid(resolution: float,
                          target_ws: gpd.GeoDataFrame) -> Tuple[xr.Dataset,
                                                                list,
                                                                np.ndarray,
                                                                np.ndarray]:
    """_summary_

    Args:
        resolution (float): _description_
        target_ws (gpd.GeoDataFrame): _description_

    Returns:
        Tuple[xr.DataArray, list]: _description_
    """
    # choose extenet
    min_LON, max_LON, min_LAT, max_LAT = find_extent(
        ws=target_ws['geometry'].values[0])
    # generate lats and lons for LISFLOOD grid
    lons = np.arange(start=min_LON,
                     stop=max_LON,
                     step=resolution)
    lats = np.arange(start=min_LAT,
                     stop=max_LAT,
                     step=resolution)
    # simulate grid
    polygons = list()
    for j in range(lats.size):
        for i in range(lons.size):
            # h = half of resolution
            # phi rotation angle
            polygons.append(Polygon(getSquareVertices(mm=(lons[i],
                                                          lats[j]),
                                                      h=resolution/2,
                                                      phi=0)))
    # create geodataframe from each polygon from emulation
    polygons = [create_GDF(poly) for poly in polygons]
    # overwrite with new lons, lats from polygon grid
    lons = np.unique([item for sublist in
                      [poly['geometry'].values[0].centroid.xy[0]
                       for poly in polygons]
                      for item in sublist])
    lats = np.unique([item for sublist in
                      [poly['geometry'].values[0].centroid.xy[1]
                       for poly in polygons]
                      for item in sublist])

    # lons, lats = np.meshgrid(lons, lats)

    vals = np.ones((lats.shape[0], lons.shape[0]))
    name = 'mask'

    mask_map = xr.Dataset(data_vars={f'{name}': (['x', 'y'], vals)},
                          coords=dict(lat=(['x'], lats),
                                      lon=(['y'], lons)),
                          attrs=dict(
        description="Area mask map 1 arcmin"))

    return (mask_map, polygons, lons, lats)


def lisflood_val_grid(grid_polygons: list,
                      val_name: str,
                      vals: np.ndarray,
                      desc: str) -> xr.Dataset:
    """_summary_

    Args:
        grid_polygons (list): _description_
        vals (np.ndarray): _description_
        desc (str): _description_

    Returns:
        xr.DataArray: _description_
    """
    lons = np.unique([item for sublist in
                      [poly['geometry'].values[0].centroid.xy[0]
                       for poly in grid_polygons]
                      for item in sublist])
    lats = np.unique([item for sublist in
                      [poly['geometry'].values[0].centroid.xy[1]
                       for poly in grid_polygons]
                      for item in sublist])

    val_map = xr.Dataset(data_vars={f'{val_name}': (['x', 'y'], vals)},
                         coords=dict(lat=(['x'], lats),
                                     lon=(['y'], lons)),
                         attrs=dict(
        description=f"{desc}"))

    return val_map


def select_NC_by_extent(nc, shape):
    """
    
    select net_cdf by extent of given shape
    
    return masked net_cdf
    
    """
    if 'x' in nc.dims:
        nc = nc.rename({'y': 'lat', 'x': 'lon'})

    # find biggest polygon
    big_shape = select_big_from_MP(WS_geometry=shape)

    # find extent coordinates
    min_LON, max_LON, min_LAT, max_LAT = find_extent(ws=big_shape)

    # select nc inside of extent
    masked_nc = nc.where(
        nc.lat >= min_LAT, drop=True).where(
        nc.lat <= max_LAT, drop=True).where(
        nc.lon >= min_LON, drop=True).where(
        nc.lon <= max_LON, drop=True)
    masked_nc = masked_nc.chunk(chunks='auto')
    return masked_nc


def interpolate_XR(ds, interp_step):
    """
    Interpolate dataset with given grad step
    """
    new_lon = np.linspace(ds.x[0], ds.x[-1],
                          num=int((ds.x.max()-ds.x.min())//interp_step))

    new_lat = np.linspace(ds.y[0], ds.y[-1],
                          num=int((ds.y.max()-ds.y.min())//interp_step))

    dsi = ds.interp(y=new_lat, x=new_lon, method='linear')

    return dsi
