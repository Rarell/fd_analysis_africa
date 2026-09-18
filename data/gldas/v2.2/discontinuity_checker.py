import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

from datetime import datetime, timedelta

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

# Made by Claude
def pettitt_test(x):
    """Pettitt's test for a single change point. Returns (index, K, approx p-value)."""

    # Force x into an array
    x = np.asarray(x)
    n = len(x)

    # Create an n by n matrix of +/- 1 depending on the sign of x
    sign_matrix = np.sign(x[:, None] - x[None, :])

    # Sum the signs at all change points
    U = np.array([np.nansum(sign_matrix[:t, t:]) for t in range(1, n)])

    # Calculate the most probable change point
    K = np.nanmax(np.abs(U))

    # 0-indexed position of the break within the series
    tau = np.argmax(np.abs(U)) + 1  

    # Pettitt (1979) approximation of the p-value; the null hypothesis is that the dataset is equal 
    # (so statistical significance means there is a significant break point)
    p = 2 * np.exp(-6 * K**2 / (n**3 + n**2)) 

    return tau, K, p

if __name__ == '__main__':
    # List of variables to check
    variables = {
        # 'temperature': 'temp',
        # 'precipitation': 'precip', 
        # 'specific_humidity': 'q',
        # 'pressure': 'pres', 
        # 'wind_speed': 'wspd', 
        # 'evaporation': 'evap', 
        # 'potential_evaporation': 'pevap', 
        'soil_moisture_root_zone': 'soilm',
    }

    alpha = 0.05

    model_change_years = [2003, 2004]

    # Load the land-sea mask
    with Dataset('../aridity_mask.nc', 'r') as nc:
        mask = nc.variables['aim'][0,:,:]
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
    
    for variable, sname in variables.items():
        if variable == 'specific_humidity':
            folder = 'moisture_surface'
        elif variable == 'soil_moisture_root_zone':
            folder = 'liquid_vsm'
        else:
            folder = variable

        # Collect the filename and variable short name
        filename = 'africa_gldas.%s.daily'%variable

        # Load the datasets to test
        with Dataset('%s/%s_%04d.nc'%(folder, filename, 2003), 'r') as nc:
            data1 = nc.variables[sname][:]

        with Dataset('%s/%s_%04d.nc'%(folder, filename, 2004), 'r') as nc:
            data2 = nc.variables[sname][:]

        data = np.concatenate([data1, data2], axis = 0)

        T, I, J = data.shape

        taus = np.zeros((I, J)) * np.nan
        Ks = np.zeros((I, J)) * np.nan
        p_values = np.ones((I, J)) * np.nan

        # p_values = mask * np.random.randint(0, 721, size = (mask.shape))

        for i in tqdm(range(I)):
            for j in range(J):
                if mask[i,j] == 0:
                    continue

                taus[i,j], Ks[i,j], p_values[i,j] = pettitt_test(data[:,i,j])

        total_discontinuities = np.nansum(p_values < alpha)
        if total_discontinuities == 0:
            print(f'No discontinuities were found in {variable}')
        else:
            Ks_sig = np.where(p_values < alpha, Ks, np.nan)
            taus_sig = np.where(p_values < alpha, taus, np.nan)
            print(f'The total number of significant discontinuities in {variable} is: {total_discontinuities}')
            print(f'Average K (most probable change point) for discontinuities: {np.nanmean(Ks_sig)} and {np.nanmean(taus_sig)} (most common index value)')

        # Plot the values

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
        fig, ax = plt.subplots(figsize = [24, 24], nrows = 1, ncols = 1, 
                            subplot_kw = {'projection': proj})

        # Add coastlines and country borders
        ax.coastlines(edgecolor = 'black', zorder = 3)
        ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        ax.set_xticks(LonLabel, crs = proj)
        ax.set_yticks(LatLabel, crs = proj)

        ax.set_yticklabels(LatLabel, fontsize = 22)
        ax.set_xticklabels(LonLabel, fontsize = 22)

        cs = ax.pcolormesh(lon, lat, p_values < 0.05, transform = proj, zorder = 1)

        # ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

        # Save the figure
        plt.savefig('%s_discontinuities_map.png'%variable, bbox_inches = 'tight')
        plt.show(block = False)
        plt.close()

        # Initialize the figure for K
        fig, ax = plt.subplots(figsize = [24, 24], nrows = 1, ncols = 1, 
                            subplot_kw = {'projection': proj})

        # Add coastlines and country borders
        ax.coastlines(edgecolor = 'black', zorder = 3)
        ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        ax.set_xticks(LonLabel, crs = proj)
        ax.set_yticks(LatLabel, crs = proj)

        ax.set_yticklabels(LatLabel, fontsize = 22)
        ax.set_xticklabels(LonLabel, fontsize = 22)

        cs = ax.pcolormesh(lon, lat, Ks, transform = proj, zorder = 1)

        # ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

        # Make the colorbar
        cbax = fig.add_axes([0.125, 0.0885, 0.780, 0.030])
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, extend = 'both', orientation = 'horizontal')

        # Make the colorbar label
        cbar.ax.set_xlabel('Strength of V2.2 Dominance over V2.0', fontsize = 22)

        for i in cbar.ax.xaxis.get_ticklabels():
            i.set_size(22)

        # Save the figure
        plt.savefig('%s_Ks_map.png'%variable, bbox_inches = 'tight')
        plt.show(block = False)
        plt.close()

        offset = (datetime(2003,1,1) - datetime(1970,1,1)).days
        taus = taus + offset

        # Make the colorbar for the change points
        cmin = 0 + offset; cmax = 721 + offset; cint = (cmax - cmin)/24 # Interval of 24; one for each month
        clev = np.round(np.arange(cmin, cmax + cint, cint))
        nlevs = len(clev)
        cmap = plt.get_cmap(name = 'viridis', lut = nlevs)

        # Initialize the figure for taus
        fig, ax = plt.subplots(figsize = [24, 24], nrows = 1, ncols = 1, 
                            subplot_kw = {'projection': proj})

        # Add coastlines and country borders
        ax.coastlines(edgecolor = 'black', zorder = 3)
        ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        ax.set_xticks(LonLabel, crs = proj)
        ax.set_yticks(LatLabel, crs = proj)

        ax.set_yticklabels(LatLabel, fontsize = 22)
        ax.set_xticklabels(LonLabel, fontsize = 22)

        cs = ax.pcolormesh(lon, lat, taus, transform = proj, vmin = cmin, vmax = cmax, cmap = cmap, zorder = 1)

        # Make the colorbar
        cbax = fig.add_axes([0.125, 0.0885, 0.780, 0.030])
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, extend = 'both', cmap = cmap, orientation = 'horizontal')

        dates = np.array([datetime(2003,1,1) + timedelta(days = day) for day in range(cmax)])

        # cbar.ax.set_xticks(mdates.date2num(dates[::24]))

        cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))

        # Make the colorbar label
        cbar.ax.set_xlabel('Change Point', fontsize = 22)

        for i in cbar.ax.xaxis.get_ticklabels():
            i.set_size(22)

        # ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

        # Save the figure
        plt.savefig('%s_taus_map.png'%variable, bbox_inches = 'tight')
        plt.show(block = False)
        plt.close()



# tau, K, p = pettitt_test(combined.values)
# detected_break_date = combined.index[tau]
# print(f"Detected break at: {detected_break_date} (K={K}, p={p:.4f})")
# print(f"Your suspected break: {break_date}")