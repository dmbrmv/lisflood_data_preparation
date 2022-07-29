from osgeo import gdal
from pcraster import (setclone, readmap, lddcreate,
                      pcr_as_numpy, slope, accuflux)
from .nc_processing import lisflood_val_grid
import numpy as np


def ldd_calculation(src_filename,
                    dst_fpath,
                    dst_fname,
                    ot,
                    VS,
                    polygons):

    dst_ds = None
    src_ds = None
    dem = None
    flow_dir = None

    src_ds = gdal.Open(src_filename)

    dst_file = f'{dst_fpath}/{dst_fname}.map'

    dst_ds = gdal.Translate(dst_file,
                            src_ds,
                            format='PCRaster',
                            outputType=ot,
                            metadataOptions=VS)
    setclone(dst_file)

    dem = readmap(dst_file)

    flow_dir = lddcreate(dem, 1e31, 1e31, 1e31, 1e31)

    # return ldd

    dir_vals = pcr_as_numpy(flow_dir).astype('byte')

    ldd_grid = np.select(condlist=[dir_vals == i for i in [-1, 0]],
                         choicelist=[5, 5],
                         default=dir_vals)
    ldd = lisflood_val_grid(grid_polygons=polygons,
                            val_name='ldd',
                            vals=ldd_grid,
                            desc='Local drain direction map')
    dst_ds = None
    src_ds = None
    dem = None
    flow_dir = None

    return ldd


def slope_calculation(src_filename,
                      dst_fpath,
                      dst_fname,
                      ot,
                      VS,
                      polygons):

    src_ds = gdal.Open(src_filename)

    dst_file = f'{dst_fpath}/{dst_fname}.map'

    dst_ds = gdal.Translate(dst_file,
                            src_ds,
                            format='PCRaster',
                            outputType=ot,
                            metadataOptions=VS)
    dst_ds = None
    src_ds = None

    setclone(dst_file)

    dem = readmap(dst_file)

    gradient = slope(dem)

    slope_vals = pcr_as_numpy(gradient).astype(np.float32)

    slope_ds = lisflood_val_grid(grid_polygons=polygons,
                                 val_name='slope',
                                 vals=slope_vals,
                                 desc='Gradient map')
    return slope_ds


def upstr_area_calculation(src_filename,
                           dst_fpath,
                           dst_fname,
                           ot,
                           VS):

    src_ds = gdal.Open(src_filename)

    dst_file = f'{dst_fpath}/{dst_fname}.map'

    dst_ds = gdal.Translate(dst_file,
                            src_ds,
                            format='PCRaster',
                            outputType=ot,
                            metadataOptions=VS)
    dst_ds = None
    src_ds = None
    setclone(dst_file)

    dem = readmap(dst_file)

    flow_dir = lddcreate(dem, 1e31, 1e31, 1e31, 1e31)

    upst_area = accuflux(flow_dir, 2)

    up_area_vals = pcr_as_numpy(upst_area).astype(np.float32)

    return up_area_vals
