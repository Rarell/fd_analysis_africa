import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import colorbar as mcolorbar
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader

from utils import subset_data

from netCDF4 import Dataset

if __name__ == '__main__':
    path = '../../gldas'
    fn = 'aridity_mask.nc'

    with Dataset('%s/%s'%(path, fn), 'r') as nc:
        desc = nc.description
        lat = nc.variables['latitude'][:]
        lon = nc.variables['longitude'][:]
        mask = nc.variables['aim'][:]

    # Subset the mask
    lon = np.where(lon < 0, lon + 360, lon)
    mask_sub, lat_sub, lon_sub = subset_data(mask, lat, lon, subset = 'africa')
    T, I, J = mask_sub.shape

    # Make the lat and lon 2D
    lon_sub, lat_sub = np.meshgrid(lon_sub, lat_sub)

    # Write the subset
    # with Dataset('../data/gldas/aridity_mask.nc', 'w', format = 'NETCDF4') as nc:
    #     # Write a description for the .nc file
    #     nc.description = desc

    #     # Create the spatial and temporal dimensions
    #     nc.createDimension('x', size = I)
    #     nc.createDimension('y', size = J)
    #     nc.createDimension('time', size = T)
        
    #     # Create the lat and lon variables       
    #     nc.createVariable('lat', lat_sub.dtype, ('x', 'y'))
    #     nc.createVariable('lon', lon_sub.dtype, ('x', 'y'))
        
    #     nc.variables['lat'][:,:] = lat_sub[:,:]
    #     nc.variables['lon'][:,:] = lon_sub[:,:]
        
    #     # Create the main variable
    #     nc.createVariable('aim', mask_sub.dtype, ('time', 'x', 'y'))
    #     nc.variables['aim'][:,:,:] = mask_sub[:,:,:]

    lon, lat = np.meshgrid(lon, lat)

    # Make a test figure of the mask

    # Lonitude and latitude tick information
    lat_int = 30
    lon_int = 40
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    # Projection information
    proj = ccrs.PlateCarree()

    # Initialize the figure
    fig = plt.figure(figsize = [20,30])
    ax = fig.add_subplot(1,1,1, projection = proj)

    # Add ocean features
    # ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Add coastlines and country borders
    ax.coastlines(edgecolor = 'black', zorder = 3)
    ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    ax.pcolormesh(lon_sub, lat_sub, mask_sub[0,:,:], vmin = 0, vmax = 1, transform = proj, zorder = 1)

    ax.set_yticklabels(LatLabel, fontsize = 26)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    ax.set_xticklabels(LonLabel, fontsize = 26)
    ax.xaxis.set_major_formatter(LonFormatter)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = proj)
    ax.set_yticks(LatLabel, crs = proj)

     # Set the map extent
    ax.set_extent([335, 53, np.nanmin(lat_sub), np.nanmax(lat_sub)])

    # Save the figure
    plt.savefig('gldas_mask.png', bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()