'''Make some quick, one panel maps to examine average
raw and computed variables.
'''

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

from netCDF4 import Dataset
import pickle

from statistics_calculations import least_squares

# Domain for Africa
lower_lat = -35; upper_lat = 35
lower_lon = 335; upper_lon = 53

# Function to make a map
def make_map(
        data, 
        lat, 
        lon, 
        title = 'title', 
        cbar_label = 'variable', 
        extend = 'both', 
        cmin = 0, 
        cmax = 1, 
        path = './', 
        savename = 'tmp.png'
        ) -> None:
    '''
    Make a simple map

    Inputs:
    :param data: Data to be plotted
    :Param lat, lon: Latitude and longitude values for data
    :param title: Title of the plot
    :param cbar_label: Label for the colorbar
    :param extend: The extend of the colorbar
    :param cmin, cmax: Minimum and maximum values of the colorbar/plotting
    :param path, savename: Path to and savename of the figure name
    '''

    data[data == 0] = np.nan

    # Colorbar information
    cint = (cmax - cmin)/100
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap = plt.get_cmap(name = 'BrBG_r', lut = nlevs)

    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    # Projection information
    proj = ccrs.PlateCarree()

    # Initialize the figure
    fig, ax = plt.subplots(figsize = [12, 12], nrows = 1, ncols = 1, 
                            subplot_kw = {'projection': proj})

    # Make the title
    ax.set_title(title, size = 22)

    # Add ocean features
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Add coastlines and country borders
    ax.coastlines(edgecolor = 'black', zorder = 3)
    ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = proj)
    ax.set_yticks(LatLabel, crs = proj)

    ax.set_yticklabels(LatLabel, fontsize = 22)
    ax.yaxis.set_major_formatter(LatFormatter)

    ax.set_xticklabels(LonLabel, fontsize = 22)
    ax.xaxis.set_major_formatter(LonFormatter)

    # Plot the data
    cs = ax.pcolormesh(lon, lat, data, 
                       vmin = cmin, vmax = cmax, cmap = cmap, 
                       transform = proj, zorder = 1)

    # Set the map extent
    ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    # Set the colorbar size and location
    cbax = fig.add_axes([0.150, 0.0485, 0.72, 0.020])

    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = extend, orientation = 'horizontal')

    # Make the colorbar label
    cbar.ax.set_xlabel(cbar_label, fontsize = 22)

    # Set the colorbar tick size
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(18)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


# Make one panel plot of PET, ET, and frequency trends (ERA5) to 
# investigate and show off the hole in central Africa

if __name__ == '__main__':
    path = '../'
    years = np.arange(1979, 2024+1)

    # Load ET
    et = []
    for year in years:
        with Dataset('%s/data/era5/evaporation/africa_evaporation_%04d.nc'%(path, year), 'r') as nc:
            et.append(nc.variables['e'][:])
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
    
    et = np.concatenate(et, axis = 0)

    # Fix a longitude issue with ERA5
    lon_ind = np.where(lon[0,:] > 330)[0]
    lon_tmp = lon[:,lon_ind]
    lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)

    tmp = et[:,:,lon_ind]
    et = np.concatenate([tmp, et[:,:,:lon_ind[0]]], axis = -1)

    # Plot time averaged ET
    make_map(
        np.nanmean(et, axis = 0), 
        lat, 
        lon, 
        title = 'Average Evaporation in Time', 
        cbar_label = 'Evaporation [m]', 
        extend = 'both', 
        cmin = -0.005, 
        cmax = 0.0, 
        path = './', 
        savename = 'era5_et_average.png',
    )

    # Load PET data
    pet = []
    for year in years:
        with Dataset('%s/data/era5/potential_evaporation/africa_potential_evaporation_%04d.nc'%(path, year), 'r') as nc:
            pet.append(nc.variables['pev'][:])
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
    
    pet = np.concatenate(pet, axis = 0)

    # Fix the ERA5 longitude issue
    lon_ind = np.where(lon[0,:] > 330)[0]
    lon_tmp = lon[:,lon_ind]
    lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)

    tmp = pet[:,:,lon_ind]
    pet = np.concatenate([tmp, pet[:,:,:lon_ind[0]]], axis = -1)

    # Plot the time averaged PET
    make_map(
        np.nanmean(pet, axis = 0), 
        lat, 
        lon, 
        title = 'Average Potential Evaporation in Time', 
        cbar_label = 'Potential Evaporation [m]', 
        extend = 'both', 
        cmin = -0.01, 
        cmax = np.nanmax(pet), 
        path = './', 
        savename = 'era5_pet_average.png',
    )

    # Load SESR data
    sesr = []
    for year in years:
        with Dataset('%s/data/era5/fd_indices/africa_sesr_%04d.nc'%(path, year), 'r') as nc:
            sesr.append(nc.variables['sesr'][:])
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
    
    sesr = np.concatenate(sesr, axis = 0)

    # Fix longitude issue with ERA5
    lon_ind = np.where(lon[0,:] > 330)[0]
    lon_tmp = lon[:,lon_ind]
    lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)

    tmp = sesr[:,:,lon_ind]
    sesr = np.concatenate([tmp, sesr[:,:,:lon_ind[0]]], axis = -1)

    # Plot time sum of ET (since the average would show up as zeros)
    make_map(
        np.nansum(sesr, axis = 0), 
        lat, 
        lon, 
        title = 'Accumulated SESR in Time', 
        cbar_label = 'SESR [unitless]', 
        extend = 'both', 
        cmin = np.nanmin(sesr), 
        cmax = np.nanmax(sesr), 
        path = './', 
        savename = 'era5_sesr_accum.png',
    )

    # Load SESR FD characteristics
    filename = '%s/data/era5/sesr_fd_characteristics_by_year_all_level_rz.pkl'%path
    with open(filename, 'rb') as f:
        characteristics = pickle.load(f)

    # Collect the frequency characteristics
    freq = characteristics['freq']

    # Fixing longitude issue
    tmp = freq[:,:,lon_ind]
    freq = np.concatenate([tmp, freq[:,:,:lon_ind[0]]], axis = -1)

    T, I, J = freq.shape

    # Collect the slope of the FD frequency change in time
    slope, _, _ = least_squares(years, freq.reshape(T, I*J))

    # freq = np.where(freq <= 0, np.nan, freq)

    # Plot total SESR FD frequency
    make_map(
        # slope.reshape(I, J), 
        np.nansum(freq, axis = 0),
        lat, 
        lon, 
        title = 'Evaporative Stress Based FD Frequency', 
        cbar_label = 'FD Frequency [# of Events]', 
        extend = 'max', 
        cmin = 0, 
        cmax = np.nanmax(np.nansum(freq, axis = 0)), 
        # cmin = np.nanmin(slope), 
        # cmax = np.nanmax(slope), 
        path = './', 
        savename = 'era5_sesr_fd_freq_average.png',
    )