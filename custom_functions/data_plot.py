
from typing import Optional

from shapely.geometry import Polygon, Point
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .geo_processing import find_extent


def plot_grids(grid_1: xr.Dataset,
               grid_3: xr.Dataset,
               target_ws: gpd.GeoDataFrame,
               map_name: str,
               grid_val_name: str,
               poly_1: Optional[list] = [],
               poly_3: Optional[list] = [],
               ) -> None:
    """_summary_

    Args:
        grid_1 (xr.Dataset): _description_
        grid_3 (xr.Dataset): _description_
        map_name (str): _description_
        poly_1 (Optional[list], optional): _description_. Defaults to [].
        poly_3 (Optional[list], optional): _description_. Defaults to [].
    """

    aea_crs = ccrs.AlbersEqualArea(central_longitude=100,
                                   standard_parallels=(50, 70),
                                   central_latitude=56,
                                   false_easting=0,
                                   false_northing=0)

    labels = ('1 arcmin grid', '3 arcmin grid')
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            subplot_kw={'projection': aea_crs},
                            figsize=(15, 7))

    fig.suptitle(f"{map_name}", fontsize=24, y=1)
    # Add the colorbar axes anywhere in the figure. Its position will be
    # re-calculated at each figure resize.
    min_LON, max_LON, min_LAT, max_LAT = find_extent(
        target_ws['geometry'].values[0])

    for grid, poly, label, ax in zip((grid_1, grid_3),
                                     (poly_1, poly_3),
                                     labels,
                                     axs.ravel()):
        # W, E, S, N
        ax.set_extent([min_LON, max_LON, min_LAT, max_LAT],
                      crs=ccrs.PlateCarree())
        lon_img, lat_img = np.meshgrid(grid['lon'], grid['lat'])

        img = ax.pcolor(lon_img,
                        lat_img,
                        grid[f'{grid_val_name}'],
                        shading='auto',
                        transform=ccrs.PlateCarree())
        if poly:
            if isinstance(poly[0].geometry.values[0], Polygon):
                ax.add_geometries([inter['geometry'].values[0]
                                   for inter in poly],
                                  crs=ccrs.PlateCarree(),
                                  facecolor="None",
                                  edgecolor='cyan',
                                  linewidth=2)
            if isinstance(poly[0].geometry.values[0], Point):
                ax.scatter([point.geometry.values[0].x for point in poly],
                           [point.geometry.values[0].y for point in poly],
                           transform=ccrs.PlateCarree())

        ax.add_geometries(target_ws['geometry'],
                          crs=ccrs.PlateCarree(),
                          facecolor="None",
                          edgecolor='red',
                          linewidth=3)

        plt.colorbar(img, ax=ax,
                     shrink=.8, pad=0.1)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')

        gl.top_labels = False
        gl.left_labels = False
        # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])

        ax.set_title(f'{label}')
        ax.patch.set_alpha(0)

    plt.tight_layout()
    plt.show()