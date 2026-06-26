'''Provides functions to create figures
and visualize results of FD analysis
'''

import numpy as np
from typing import Tuple, Union
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

# Domain for Africa
lower_lat = -35; upper_lat = 35
lower_lon = 335; upper_lon = 53
# lower_lat = -85; upper_lat = 85
# lower_lon = -180; upper_lon = 180

def make_statistics_maps(
        data, 
        lat, 
        lon, 
        statistic, 
        fd_types, 
        times, 
        cmin: Union[int, float] = 0, 
        cmax: Union[int, float] = 50, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    """
    Make a set of maps for different FD statistics (frequency, intensity, or duration), or all of them together

    Inputs:
    :param data: Lists of the different arrays being plotted (first list are the columns, second list is the rows)
    :param lat, lon: Latitudes and longitudes for the datasets
    :param statistic: Name of the statistic/characteristic being plotted, 
                      or "characteristics" when plotting frequency, duration, and severity together
    :param fd_types: List of FD identification methods being plotted
    :param times: List of times (annual, JJA, SON, etc.) that each column displays (this is the title for each column)
    :param cmin, cmax: Maximum and minimum values for colorbar (only used when statistic != "characteristics")
    :param path, savename: Path to save directory and name to save the figure as
    """

    # Set colorbar information 
    if statistic == 'characteristics':
        # For characteristics, assumes data is given in order as frequency, duration, and severity
        # And multiple colorbars need to be setup
        cmins = [0, 0, -1]
        cmaxes = [50, 50, 0]
        cints = [(cmax - cmin)/20 for (cmax, cmin) in zip(cmaxes, cmins)]
        clevs = [np.arange(cmin, cmax + cint, cint) for (cmax, cmin, cint) in zip(cmaxes, cmins, cints)]
        nlevs = [len(clev) for clev in clevs]
        cmaps = [plt.get_cmap(name = 'Spectral_r', lut = nlevs[0]), 
                 plt.get_cmap(name = 'Spectral_r', lut = nlevs[1]), 
                 plt.get_cmap(name = 'Spectral', lut = nlevs[2])]

        # Also define/initialize the mappables for the colorbar
        cses = [None, None, None]

    else:
        # Setup a single colormap for a given characteristic
        cint = (cmax - cmin)/20
        clevs = np.arange(cmin, cmax + cint, cint)
        nlevs = len(clevs)
        # Higher severity is more negative, so a reverse of normal is needed
        if statistic == 'severity':
            cmap  = plt.get_cmap(name = 'Spectral', lut = nlevs)
        else:
            cmap  = plt.get_cmap(name = 'Spectral_r', lut = nlevs)

    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    # Empty formatters
    y_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''})
    x_formatter = cticker.LatitudeFormatter(cardinal_labels = {'west': '', 'east': ''})

    # Projection information
    proj = ccrs.PlateCarree()

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [24, 24], nrows = 3, ncols = 3, 
                             subplot_kw = {'projection': proj})
    plt.subplots_adjust(wspace = -0.12, hspace = 0.08)

    # Data is organized as data['time']['fd_type']
    # Loop through rows
    for i in range(3):
        # Loop through columns
        for j in range(3):
            # Set the title for the first row
            if i == 0:
                axes[i,j].set_title(times[j], size = 22)

            # Add ocean features
            axes[i,j].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

            # Add coastlines and country borders
            axes[i,j].coastlines(edgecolor = 'black', zorder = 3)
            axes[i,j].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

            # Adjust the ticks
            axes[i,j].set_xticks(LonLabel, crs = proj)
            axes[i,j].set_yticks(LatLabel, crs = proj)

            # Add yticks for the first column, else do not add ticks
            if (j == 0):
                axes[i,j].set_yticklabels(LatLabel, fontsize = 22)
                axes[i,j].yaxis.set_major_formatter(LatFormatter)
            else:
                axes[i,j].yaxis.set_major_formatter(y_formatter)
                axes[i,j].set_yticklabels('', fontsize = 0)

            # Add xticks on the last row, else add no ticks
            if (i == 2):
                axes[i,j].set_xticklabels(LonLabel, fontsize = 22)
                axes[i,j].xaxis.set_major_formatter(LonFormatter)
            else:
                axes[i,j].xaxis.set_major_formatter(x_formatter)
                axes[i,j].set_xticklabels('', fontsize = 0)

            # Set y labels for the first column
            if j == 0:
                axes[i,j].set_ylabel(fd_types[i].upper(), fontsize = 22)
            
            # Plot the data
            if statistic == 'characteristics':
                cmap = cmaps[j]
                cmin = cmins[j]; cmax = cmaxes[j]

            cs = axes[i,j].pcolormesh(lon, lat, data[j][i], vmin = cmin, vmax = cmax,
                                      cmap = cmap, transform = proj, zorder = 1)

            # Save the color mesh and scheme if necessary
            if statistic == 'characteristics':
                cses[j] = cs
            
            # Set the map extent
            axes[i,j].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    # Set the colorbar size and location
    if statistic == 'characteristics':
        # For plotting all characteristics, multiple colorbars are needed
        cbaxes = [fig.add_axes([0.150, 0.0685, 0.22, 0.020]),
                  fig.add_axes([0.400, 0.0685, 0.22, 0.020]),
                  fig.add_axes([0.650, 0.0685, 0.22, 0.020])]
        extends = ['max', 'max', 'min']

        # Make the colorbars
        for n in range(len(cbaxes)):
            cbar = mcolorbar.Colorbar(cbaxes[n], mappable = cses[n], cmap = cmaps[n], extend = extends[n], orientation = 'horizontal')

            # Make the colorbar label
            cbar.ax.set_xlabel(times[n], fontsize = 22)

            # Set the colorbar tick size
            for i in cbar.ax.xaxis.get_ticklabels():
                i.set_size(22)

    else:
        # For plotting a single characteristic, set the size and location for a single colorbar
        cbax = fig.add_axes([0.150, 0.0685, 0.72, 0.020])

        # Different extent is needed for severity since it worsening conditions is lower values
        if statistic == 'severity':
            extend = 'min'
            label = 'Normalized Severity'
        else:
            extend = 'max'
            label = statistic.title()

        # Make the colorbar
        cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = extend, orientation = 'horizontal')

        # Make the colorbar label
        cbar.ax.set_xlabel(label, fontsize = 22)

        # Set the colorbar tick size
        for i in cbar.ax.xaxis.get_ticklabels():
            i.set_size(22)
            
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def make_trend_maps(
        data, 
        lat, 
        lon, 
        statistic, 
        fd_types, 
        times, 
        sig: Union[float, np.ndarray] = None, 
        cmin: Union[int, float] = -1, 
        cmax: Union[int, float] = 1, 
        significance_plots: bool = False, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    """
    Make a set of maps for trends in FD characteristics (of frequency, intensity, or duration and their statistical significance)
    
    Note: This is similar make_statistics_map except includes code significance results

    Inputs:
    :param data: Lists of the different arrays being plotted (first list are the columns, second list is the rows)
    :param lat, lon: Latitudes and longitudes for the datasets
    :param statistic: Name of the statistic/characteristic being plotted, 
                      or "characteristics" when plotting frequency, duration, and severity together
    :param fd_types: List of FD identification methods being plotted
    :param times: List of times (annual, JJA, SON, etc.) that each column displays (this is the title for each column)
    :param sig: P-values of the trends (same list structure as data)
    :param cmin, cmax: Maximum and minimum values for colorbar (only used when statistic != "characteristics")
    :param significance_plots: Bool. Gives whether statistical significance is being plotted
    :param path, savename: Path to save directory and name to save the figure as
    """

    alpha = 0.05
    units = {'frequency': 'Events / Grid Point / Year', 
             'duration': 'Days / Year', 
             'severity': 'Unitless / Year'}

    # Set colorbar information 
    if statistic == 'characteristics':
        # For characteristics, assumes data is given in order as frequency, duration, and severity
        # And multiple colorbars need to be setup
        cmins = [-0.06, -0.5, -0.0015]
        cmaxes = [0.06, 0.5, 0.0015]
        cints = [(cmax - cmin)/20 for (cmax, cmin) in zip(cmaxes, cmins)]
        clevs = [np.arange(cmin, cmax + cint, cint) for (cmax, cmin, cint) in zip(cmaxes, cmins, cints)]
        nlevs = [len(clev) for clev in clevs]
       
        if significance_plots:
            # Special set of colorboxes for displaying statistical significance
            # cmap = plt.get_cmap(name = 'Paired', lut = nlevs)
            cmap = mcolors.ListedColormap(['#FF796C', '#DC143C', '#7BC8F6', '#0000FF'])
        else:
            # Make multiple colorbars
            cmaps = [plt.get_cmap(name = 'BrBG_r', lut = nlevs[0]), 
                     plt.get_cmap(name = 'BrBG_r', lut = nlevs[1]), 
                     plt.get_cmap(name = 'BrBG', lut = nlevs[2])]

        # Also define/initialize the mappables for the colorbar
        cses = [None, None, None]
    
    else:
        # When only 1 characteristic is plotted, use arguments to define the colorbar
        cint = (cmax - cmin)/20
        clevs = np.arange(cmin, cmax + cint, cint)
        nlevs = len(clevs)
        if significance_plots:
            # Special set of colorboxes for displaying statistical significance
            # cmap = plt.get_cmap(name = 'Paired', lut = nlevs)
            cmap = mcolors.ListedColormap(['#FF796C', '#DC143C', '#7BC8F6', '#0000FF'])
        else:
            # Note severe colorbar is reversed since decreasing values indicate worse severity
            if statistic == 'severity':
                cmap = plt.get_cmap(name = 'BrBG', lut = nlevs)
            else:
                cmap = plt.get_cmap(name = 'BrBG_r', lut = nlevs)

    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    # Empty formatters
    y_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''})
    x_formatter = cticker.LatitudeFormatter(cardinal_labels = {'west': '', 'east': ''})

    # Projection information
    proj = ccrs.PlateCarree()

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [24, 24], nrows = 3, ncols = 3, 
                             subplot_kw = {'projection': proj})
    plt.subplots_adjust(wspace = -0.12, hspace = 0.08)

    # Data is organized as data['time']['fd_type']
    # Loop through rows
    for i in range(3):
        # Loop through columns
        for j in range(3):
            # Set the title for the first row
            if i == 0:
                axes[i,j].set_title(times[j], size = 22)

            # Add ocean features
            axes[i,j].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

            # Add coastlines and country borders
            axes[i,j].coastlines(edgecolor = 'black', zorder = 3)
            axes[i,j].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

            # Adjust the ticks
            axes[i,j].set_xticks(LonLabel, crs = proj)
            axes[i,j].set_yticks(LatLabel, crs = proj)

            # Add yticks for the first column, else do not add ticks
            if (j == 0):
                axes[i,j].set_yticklabels(LatLabel, fontsize = 22)
                axes[i,j].yaxis.set_major_formatter(LatFormatter)
            else:
                axes[i,j].yaxis.set_major_formatter(y_formatter)
                axes[i,j].set_yticklabels('', fontsize = 0)

			# Add xticks on the last row, else add no ticks
            if (i == 2):
                axes[i,j].set_xticklabels(LonLabel, fontsize = 22)
                axes[i,j].xaxis.set_major_formatter(LonFormatter)
            else:
                axes[i,j].xaxis.set_major_formatter(x_formatter)
                axes[i,j].set_xticklabels('', fontsize = 0)

            # Set y labels for the first column
            if j == 0:
                # Also set the label
                axes[i,j].set_ylabel(fd_types[i].upper(), fontsize = 22)
            
            # Plot the data
            if significance_plots:
                # Determine areas of statistical significance
                significance_data = np.where(((sig[i][j] > (1-alpha/2)) | (sig[i][j] < (alpha/2))) & (data[i][j] > 0), 1, 0)
                significance_data = np.where(((sig[i][j] < (1-alpha/2)) & (sig[i][j] > (alpha/2))) & (data[i][j] < 0), 2, significance_data)
                significance_data = np.where(((sig[i][j] > (1-alpha/2)) | (sig[i][j] < (alpha/2))) & (data[i][j] < 0), 3, significance_data)
                # Plot the statistical significance
                cs = axes[i,j].pcolormesh(lon, lat, significance_data, vmin = 0, vmax = 3,
                                          cmap = cmap, transform = proj, zorder = 1)
            else:
                # Plot the trends
                if statistic == 'characteristics':
                    cmap = cmaps[j]
                    cmin = cmins[j]; cmax = cmaxes[j]

                cs = axes[i,j].pcolormesh(lon, lat, data[i][j], vmin = cmin, vmax = cmax,
                                          cmap = cmap, transform = proj, zorder = 1)

				# Save the color mesh and scheme if necessary
                if statistic == 'characteristics':
                    cses[j] = cs
            
            # Set the map extent
            axes[i,j].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


    # Make the colorbar if needed
    if significance_plots == False:
        # When plotting all characteristics, multiple colorbars are needed
        if statistic == 'characteristics':
            cbaxes = [fig.add_axes([0.150, 0.0685, 0.22, 0.020]),
                      fig.add_axes([0.400, 0.0685, 0.22, 0.020]),
                      fig.add_axes([0.650, 0.0685, 0.22, 0.020])]

            keys = ['frequency', 'duration', 'severity']

            # Make the colorbars
            for n in range(len(cbaxes)):
                cbar = mcolorbar.Colorbar(cbaxes[n], mappable = cses[n], cmap = cmaps[n], extend = 'both', orientation = 'horizontal')

                # Make the colorbar labels
                cbar.ax.set_xlabel('%s Trends\n[%s]'%(times[n], units[keys[n]]), fontsize = 22)

                # Frequency and severity ticks labels don't quite fit on the smaller colorbars
                # Adjust the ticks to use half the labels to better fit
                if keys[n] in ['frequency', 'severity']:
                    ticks = cbar.ax.xaxis.get_ticklabels()
                    ticks_new = []
                    # Ticks obtained from matplotlib use a special hyphen not in UTF-8; 
                    # replace it with the normal hyphen if needed
                    for n in range(len(ticks)):
                        tick = ticks[n].get_text()
                        if '−' in tick:
                            tick = tick.replace('−', '-')
                        ticks_new.append(float(tick))
                        
                    ticks_new = np.array(ticks_new)
                    
                    # Set the ticks
                    cbar.ax.xaxis.set_ticks(ticks_new[::2])

                # Set the colorbar tick size
                for i in cbar.ax.xaxis.get_ticklabels():
                    i.set_size(22)
        else:
            # Set a single colorbar size and location for a characteristic
            cbax = fig.add_axes([0.150, 0.0685, 0.72, 0.020])

			# Special label is needed for severity since it uses normalized severity
            if statistic == 'severity':
                extend = 'both'
                label = 'Normalized Severity Trends [%s]'%units[statistic]
            else:
                extend = 'both'
                label = '%s Trends [%s]'%(statistic.title(), units[statistic])

            cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = extend, orientation = 'horizontal')

            # Make the colorbar label
            cbar.ax.set_xlabel(label, fontsize = 22)

            # Set the colorbar tick size
            for i in cbar.ax.xaxis.get_ticklabels():
                i.set_size(22)
                
    # Statistical significance uses a set of boxes in a legend to indicate statistical significance with increasing/decreasing trends
    else:
        # Custom patches/boxes for the legend
        not_sig_pos = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap(0.0), edgecolor = 'k', label = 'Not Significant Increasing')
        sig_pos = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap(0.25), edgecolor = 'k', label = 'Significant Increasing')
        not_sig_neg = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap(0.50), edgecolor = 'k', label = 'Not Significant Decreasing')
        sig_neg = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap(0.75), edgecolor = 'k', label = 'Significant Decreasing')
        
        # Add boxes indicating significance colors
        fig.legend(handles = [not_sig_pos, sig_pos, not_sig_neg, sig_neg], bbox_to_anchor = (0.75, 0.1), frameon = False, ncols = len([not_sig_pos, sig_pos]), fontsize = 22) # loc = 'lower center',
            
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def make_correlation_maps(
        data, 
        pvals, 
        lat, 
        lon, 
        fd_types, 
        var_name, 
        index_corr: bool = False, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make a map of Pearson correlations with statistical significances

    Inputs:
    :param data: List of the different correlation arrays being plotted (assumed three datasets)
    :param pvals: P-values of the correlations (same list structure as data)
    :param lat, lon: Latitudes and longitudes for the datasets
    :param fd_types: List of FD identification methods/indices used in the correlation analyses
    :param var_name: Name of the variable the FD/FD indices were correlated with
    :param index_corr: Bool. Indicates whether the correlation is with a FD index or not
    :param path, savename: Path to save directory and name to save the figure as
    '''
    alpha = 0.05

    # Set colorbar information 
    if index_corr:
        cmax = 1.0; cmin = -1.0
    else:
        cmax = 0.5; cmin = -0.5
    cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'coolwarm', lut = nlevs)

    cmap_significance  = plt.get_cmap(name = 'Paired', lut = nlevs)

    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    # Empty formatters
    y_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''})
    x_formatter = cticker.LatitudeFormatter(cardinal_labels = {'west': '', 'east': ''})

    # Projection information
    proj = ccrs.PlateCarree()

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [16, 24], nrows = 2, ncols = 3, 
                             subplot_kw = {'projection': proj})
    plt.subplots_adjust(wspace = 0.12, hspace = -0.65)

	# Add the title
    fig.suptitle('Correlation between FD and %s'%var_name, y = 0.72, fontsize = 22)
 
    # Loop through columns
    for j in range(3):
        # Set the subtitle
        axes[0,j].set_title(fd_types[j].upper(), size = 22)

        # Add ocean features
        axes[0,j].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

        # Add coastlines and country borders
        axes[0,j].coastlines(edgecolor = 'black', zorder = 3)
        axes[0,j].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        axes[0,j].set_xticks(LonLabel, crs = proj)
        axes[0,j].set_yticks(LatLabel, crs = proj)

		# Add yticks for the first column, else do not add ticks
        if (j == 0):
            axes[0,j].set_yticklabels(LatLabel, fontsize = 22)
            axes[0,j].yaxis.set_major_formatter(LatFormatter)
        else:
            axes[0,j].yaxis.set_major_formatter(y_formatter)
            axes[0,j].set_yticklabels('', fontsize = 0)

		# Set the xticks to empty
        axes[0,j].xaxis.set_major_formatter(x_formatter)
        axes[0,j].set_xticklabels('', fontsize = 0)
        
        # Plot the correlation data
        cs_data = axes[0,j].pcolormesh(lon, lat, data[j], vmin = cmin, vmax = cmax,
                                       cmap = cmap, transform = proj, zorder = 1)
        
        # Set the map extent
        axes[0,j].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

        # Begin working on the row with the significance

        # Add ocean features
        axes[1,j].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

        # Add coastlines and country borders
        axes[1,j].coastlines(edgecolor = 'black', zorder = 3)
        axes[1,j].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        axes[1,j].set_xticks(LonLabel, crs = proj)
        axes[1,j].set_yticks(LatLabel, crs = proj)

		# Add yticks for the first column, else do not add ticks
        if (j == 0):
            axes[1,j].set_yticklabels(LatLabel, fontsize = 22)
            axes[1,j].yaxis.set_major_formatter(LatFormatter)
        else:
            axes[1,j].yaxis.set_major_formatter(y_formatter)
            axes[1,j].set_yticklabels('', fontsize = 0)

		# Set the xticks
        axes[1,j].set_xticklabels(LonLabel, fontsize = 22)
        axes[1,j].xaxis.set_major_formatter(LonFormatter)

        
        # Determine areas of statistical significance
        significance_data = np.where((pvals[j] > (1-alpha/2)) | (pvals[j] < (alpha/2)), 1, 0)

        # Plot the data
        cs_sig = axes[1,j].pcolormesh(lon, lat, significance_data, vmin = 0, vmax = 1,
                                      cmap = cmap_significance, transform = proj, zorder = 1)
        
        # Set the map extent
        axes[1,j].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

        
    # Set the colorbar size and location
    cbax = fig.add_axes([0.910, 0.5085, 0.020, 0.170])

    label = 'Correlation Coefficient\n(%s and FD)'%(var_name)

    cbar = mcolorbar.Colorbar(cbax, mappable = cs_data, cmap = cmap, orientation = 'vertical')

    # Make the colorbar label
    cbar.ax.set_ylabel(label, fontsize = 22)

    # Set the colorbar tick size
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(22)
    
    # Make the patches for statistical signicance
    not_sig = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap_significance(0.0), edgecolor = 'k', label = 'Not Significant')
    sig = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap_significance(1.0), edgecolor = 'k', label = 'Significant')
    # Add boxes indicating significance colors
    fig.legend(handles = [not_sig, sig], bbox_to_anchor = (0.75, 0.3), frameon = False, ncols = len([not_sig, sig]), fontsize = 22) # loc = 'lower center',
            
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


def make_lagged_correlation_plot(
        data, 
        pvals, 
        lags, 
        snames, 
        fd_types, 
        labels, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make a set of lagged correlation plots with significance indicated by diamonds

    Inputs:
    :param data: Directory of lagged correlations to plot (each key indicates the FD method and variable it is correlated with)
    :param pvals: P-values fo each lagged correlation value (same dictionary structure and naming as data)
    :param lags: Lag times used for determining correlations
    :param snames: List of variable names that was correlated with FD/FD indices
    :param fd_types: List of FD identification methods/indices used in the correlation analyses
    :param labels: List of labels used to label each line in the correlation plot
    :param path, savename: Path to save directory and name to save the figure as
    '''
    alpha = 0.05
    ncols = 3
    # colors = ['k', 'r', 'b']

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [28, 8], nrows = 1, ncols = ncols)

	# Loop through the columns
    for n in range(ncols):
        # Make the regression line for each variable
        for m, sname in enumerate(snames):
            # Determine lag times with statistical significance
            significance_data = [(pval > (1-alpha/2)) | (pval < (alpha/2)) for pval in pvals['%s_%s'%(fd_types[n], sname)]]
            #np.where((pvals['%s_%s'%(fd_types[n], sname)] > (1-alpha/2)) | (pvals[['%s_%s'%(fd_types[n], sname)] < (alpha/2)), 1, 0)

            # Plot the time series 
            axes[n].plot(lags, data['%s_%s'%(fd_types[n], sname)], marker = 'd', markevery = significance_data, label = labels[m])

        # Set the plot title
        axes[n].set_title(fd_types[n].upper(), fontsize = 22)

        # Add the legend
        axes[n].legend(fontsize = 16)

        # Set the axes labels
        axes[n].set_xlabel('Lagged Time [days]', fontsize = 22)
        
        # ylabel is only added on the first column
        if n == 0:
            axes[n].set_ylabel('Lagged Correlation Coefficient', fontsize = 22)

        # Set the tick size
        for i in axes[n].xaxis.get_ticklabels() + axes[n].yaxis.get_ticklabels():
            i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


def make_boxplots(
        data, 
        fd_types, 
        times, 
        statistic, 
        path = './', 
        savename = 'tmp.png'
        ) -> None:
    """
    Make a set of three box and whiskers plot of some data

    Inputs:
    :param data: Lists of the different arrays being plotted (first list are the columns, second list each data for each box and whiskers)
    :param fd_types: List of FD identification methods being plotted
    :param times: List of times (annual, JJA, SON, etc.) that each box and whiskers in a plot displays (this is the label for each box and whiskers)
    :param statistic: Name of the statistic/characteristic being plotted
    :param path, savename: Path to save directory and name to save the figure as
    """

    # Define the label
    if statistic == 'frequency':
        label = 'Events / grid point'
    elif statistic == 'duration':
        label = 'Length of Events [days]'
    elif statistic == 'severity':
        label = 'Normalized Accumulation of Index\nduring Flash Drought [unitless]'

    # Remove NaNs from distributions
    for n in range(len(data)):
        for m in range(len(data[n])):
            data[n][m] = np.delete(data[n][m], (data[n][m] == 0) | np.isnan(data[n][m]))

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [24, 8], nrows = 1, ncols = 3)
    # Severity needs some extra spacing since its values are longer (i.e., 0.001 vs 10)
    if statistic == 'severity':
        plt.subplots_adjust(wspace = 0.45)

    # Perform plot for each column
    for j in range(3):

        # Set the title
        axes[j].set_title(fd_types[j].upper(), fontsize = 22)

        # Make the box and whiskers plot
        bplots = axes[j].boxplot(
            data[j], 
            notch = True, 
            whis = (5, 95),
            sym = '', 
            vert = True, 
            bootstrap = 5000, 
            patch_artist = True, 
            tick_labels = times, 
            manage_ticks = True, 
            meanline = True, 
            showcaps = False, 
            showmeans = False,
        )

        # Set boxplot colors
        colors = [
            '#9A0EEA', # Violet
            '#E50000', # Red
            '#069AF3' # Azure
        ]

        for patch, color in zip(bplots['boxes'], colors):
            patch.set_facecolor(color)

        # Add labels
        axes[j].set_ylabel(label, fontsize = 22)

        # Set the tick size
        for i in axes[j].xaxis.get_ticklabels() + axes[j].yaxis.get_ticklabels():
            i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_variable_boxplots(
        data, 
        var_labels, 
        labels, 
        fd_type, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make a box plot for multiple variables during FD. Note each variable has multiple entries (e.g., 30 days before FD, 20 days before, etc.).

    Inputs:
    :param data: Lists of the different arrays being plotted 
                (first list has entries for variables 30 days before FD, 20 days before, etc., second list each data for each box and whiskers)
    :param var_labels: List of variable names used to label each box and whisker
    :param labels: List of labels for each entry in the first list of data
    :param fd_type: The FD identification method used to determine data
    :param path, savename: Path to save directory and name to save the figure as
    
    '''

	# Remove NaNs from the distributions
    for n in range(len(data)):
        for m in range(len(data[n])):
            data[n][m] = np.delete(data[n][m], np.isnan(data[n][m]))

    # Initialize the figure
    fig, ax = plt.subplots(figsize = [18, 8], nrows = 1, ncols = 1)

    # Set the title
    ax.set_title(fd_type.upper(), fontsize = 22)

    # Set boxplot colors
    colors = [
        '#380282', # Indigo
        '#008000', # (Dark) Green
        '#7BC8F6', # Lightblue
        '#9ACD32' # Yellowgreen
    ]

    # Make the box and whiskers plot
    bplots_total = []
    for n in range(len(data)):
        bplots = ax.boxplot(
            data[n], 
            notch = True, 
            whis = (5, 95),
            positions = [n+1.5+5*m for m in range(len(data[n]))],
            sym = '', 
            vert = True, 
            bootstrap = 1000, 
            patch_artist = True, 
            tick_labels = var_labels if n == 1 else ['' for _ in range(len(var_labels))],
            manage_ticks = True, 
            meanline = True,
            showcaps = False,
            showmeans = False,
        )

		# Set the boxplot facecolors
        for patch in bplots['boxes']:
            patch.set_facecolor(colors[n])

        bplots_total.append(bplots['boxes'][0])

        # if n == 1:
            
        # else:
        #     # ax.set_xticks([]) # Remove xticks when there is no label
            

    # Add a zero line
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 2.0)

    # Set x ticks
    ax.set_xticks(np.arange(2.5, 5*len(data[n]), 5))
    ax.set_xticklabels(var_labels, fontsize = 22)

    # Add a legend
    ax.legend(bplots_total, labels, loc = 'upper right', fontsize = 16)

    # Add labels
    ax.set_ylabel('Standardized Anomalies [unitless]', fontsize = 22)

    # Set the tick size
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_barplots(
        data, 
        labels, 
        ylabel, 
        xtick_labels, 
        bar_err: Union[float, np.ndarray] = None, 
        title: Union[float, str] = None, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make a barplot of multiple inputs

    Inputs:
    :param data: List of data with each entry being an input for the bar plot
    :param labels: List of labels for each set of bars
    :param ylabel: Label for the y-axis
    :param xtick_labels: List of labels for the x-axis
    :param bar_err: Error bars to add to each bar
    :param title: Title for the plot
    :param path, savename: Path to save directory and name to save the figure as
    '''

    # Initialize the figure
    fig, ax = plt.subplots(figsize = [14, 8], nrows = 1, ncols = 1)

	# Define colors for the bar plots
    colors = ['grey', 
              '#DC143C', # Crimson
              '#0000FF'  # Blue
              ]

	# Set the title if given
    if title is not None:
        ax.set_title(title, fontsize = 22)

    # Make the bar plots
    for n, dataset in enumerate(data):
        # Determine the width and position of the bars
        ind = np.arange(len(dataset))
        width = 0.25

		# Determine the error range of each bar if given
        if bar_err is not None:
            err = bar_err[n]
        else:
            err = None
        print(err)

		# Add the bars to the bar plot
        ax.bar(ind + n*width, dataset, width, facecolor = colors[n], alpha = 1.0, yerr = err, label = labels[n].upper())

    # Set the legend
    ax.legend(fontsize = 22)

    # Set the ticks
    ax.set_xticks(ind + width, labels = xtick_labels)

    # Set t label
    ax.set_ylabel(ylabel, fontsize = 22)
    ax.set_ylim([0, 65])

    # Set the tick size
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_errorbar_plot(
        x, 
        y, 
        yerr,
        xlabel, 
        ylabel, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make an errorbar plot

    Inputs:
    :param x, y: One dimensional x and y data being plotted
    :param yerr: Errors/uncertainty in y
    :param xlabel: Label for x-axis
    :param ylabel: Label for y-axis
    :param path, savename: Path to save directory and name to save the figure as
    '''
    
    # Formatting for the scientific notation used
    plt.ticklabel_format(axis = 'both', useMathText = True)

    # Initialize the figure
    fig, ax = plt.subplots(figsize = [14, 8], nrows = 1, ncols = 1)

    # Make the errorbar plot
    ax.errorbar(x, y, yerr = yerr, color = 'r', fmt = 'o')

    # Set the labels
    ax.set_ylabel(ylabel, fontsize = 22)
    ax.set_xlabel(xlabel, fontsize = 22)

    # Set the tick size
    ax.yaxis.get_offset_text().set_fontsize(22)
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


def timeseries_plot(
        x, 
        y, 
        slopes, 
        intercepts, 
        p_values, 
        fd_types, 
        label, 
        times, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make multiple time series plots (columns of three), with multiple lines and regression lines for each line

    Inputs:
    :param x: x values to plot against
    :param y: Lists time series to plot (first list is each line in a plot and second list is each column in the figure)
    :param slopes: Lists for the slopes for each line in y (same list format as y)
    :param intercepts: Lists of intercepts for each line in y (same list format as y)
    :param p_values: Lists of p-values for the slopes (same list format as y)
    :param fd_types: List of FD identification methods used for each plot
    :param label: Name of the variable being plotted
    :param times: List of times (annual, JJA, SON, etc.) that each column displays (this is the title for each column)
    :param path, savename: Path to save directory and name to save the figure as
    '''

	# Initialize some variables for each line
    ncols = 3
    colors = ['k', 'r', 'b']
    units = {'frequency': 'Events / Grid Point', 
             'duration': 'Days', 
             'severity': 'Unitless'}
    symbols = ['o', '^', 'd']

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [28, 8], nrows = 1, ncols = ncols)

    for n in range(ncols):
        # Make the regression lines
        for m in range(len(y)):
        	# Make the regression estimate of the time series
            yhat = slopes[m][n] * x + intercepts[m][n]

            # Plot the time series and regression line
            axes[n].plot(x, y[m][n], color = colors[m], marker = symbols[m], label = '')
            axes[n].plot(
                x, 
                yhat, 
                color = colors[m], 
                linestyle = '--', 
                marker = symbols[m], 
                label = r'%s: $\hat{y}$ = %6.4ft+%5.2f, p-value = %4.3f'%(fd_types[m].upper(), slopes[m][n], intercepts[m][n], p_values[m][n]),
            )

        # Set the plot title
        axes[n].set_title(times[n], fontsize = 22)

        # Add the legend
        axes[n].legend(fontsize = 16)

        # Set the label
        axes[n].set_xlabel('Time', fontsize = 22)
        # Set the ylabel for the first column only
        if n == 0:
            axes[n].set_ylabel('Averaged %s [%s]'%(label.title(), units[label]), fontsize = 22)

        # Set the tick size
        for i in axes[n].xaxis.get_ticklabels() + axes[n].yaxis.get_ticklabels():
            i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_scatterplots(
        x, 
        y, 
        z, 
        r, 
        pval, 
        labels, 
        suptitles, 
        slope: Union[float, np.ndarray] = None, 
        intercept: Union[float, np.ndarray] = None, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make a set of scatterplots between a set of datasets with linear regression lines between variables

    Inputs:
    :param x, y, z: Lists of datasets being plotted, all must be the same size
    :param r: Array of correlation values between each dataset, 
              formatted as [[x[0] and y[0], x[0] and z[0], y[0] and z[0]], etc.]
    :param pval: Array of p-values for each correlation (same format as r)
    :param labels: List of labels for each dataset [x, y, z]
    :param suptitles: List of subtitles to use at the top of each column in the figure
    :param slope: Array of regression slopes between each dataset (same format as r)
    :param intercept: Array of intercepts between each dataset (same format as r)
    :param path, savename: Path to save directory and name to save the figure as
    '''
    
    # Determine maximum values for setting uniform limits
    max_limit = np.nanmax([np.nanmax(x), np.nanmax(y), np.nanmax(z)])

    # Create a colorbar
    cmin = 0.0001; cmax = 300
    cint = (cmax - cmin)/200
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'rainbow', lut = nlevs)

    ncols = 3

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [24, 24], nrows = len(x), ncols = ncols)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.25)

    for i in range(len(x)):
        
        # Collect the index to correlate the current row of data with
        alt_ind = i+1
        if alt_ind >= len(x):
            alt_ind = 0

        for j in range(ncols):
            ax = axes[i,j]
            if j == 0:
                data = x
            elif j == 1:
                data = y
            else:
                data = z

			# Make the 2D density plot
            h2d = ax.hist2d(
                data[i], 
                data[alt_ind], 
                bins = 60, 
                cmin = cmin, 
                vmax = cmax, 
                edgecolor = 'face', 
                cmap = cmap, 
                rasterized = True,
            )

            # Set the ideal line
            ideal_line = np.arange(0, max_limit)
            ax.plot(ideal_line, ideal_line, color = 'grey', linestyle = '--', linewidth = 1.0)

            # Set the regression line if given
            if (slope is not None) & (intercept is not None):
                regress_line = slope[i,j] * ideal_line + intercept[i,j]
                ax.plot(ideal_line, regress_line, color = 'k', linewidth = 2.0)

            # Set limits
            ax.set_xlim([0, max_limit])
            ax.set_ylim([0, max_limit])

            # Set labels
            ax.set_xlabel(labels[i].upper(), fontsize = 22)
            ax.set_ylabel(labels[alt_ind].upper(), fontsize = 22)

            # Use the title to set the correlation and pval
            if i == 0:
                ax.set_title("%s\nPearson's r = %4.2f, p-value = %4.3f"%(suptitles[j], r[i,j], pval[i,j]), fontsize = 22)
            else:
                ax.set_title("Pearson's r = %4.2f, p-value = %4.3f"%(r[i,j], pval[i,j]), fontsize = 22)

            # Set the tick sizes
            for tick in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                tick.set_size(18)

    # Add a colorbar
    # cbax = fig.add_axes([0.91, 0.11, 0.018, 0.77]) # for vertical orientation
    cbax = fig.add_axes([0.12, 0.05, 0.78, 0.018]) # for horizontal orientation
    # Add the colorbar
    cbar = mcolorbar.Colorbar(cbax, mappable = h2d[3], extend = 'max', orientation = 'horizontal')

    # Add a colorbar label
    cbar.ax.set_xlabel('Counts', fontsize = 22)

    # Set the colorbar tick size
    for i in cbar.ax.xaxis.get_ticklabels():
        i.set_size(22)


    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_eof_plot(
        regression_patterns, 
        lat, 
        lon, 
        pcs, 
        times, 
        variable, 
        mode, 
        var_explained, 
        path: str = './', 
        savename: str = 'tmp.png'
        ) -> None:
    '''
    Make a plot of an EOF regression pattern with the PCs under it

    Inputs:
    :param regression_patterns: Two dimensional array of regression patterns (of an EOF) to plot
    :param lat, lon: Latitudes and longitudes for the regression patterns
    :param pcs: One dimensional array of PCs to plot as a time series
    :param times: Array of datetimes to plot against pcs
    :param variable: Name of the variable the EOF pattern is for
    :param mode: The mode of the EOF pattern
    :param var_explained: Percentage of the variance explained by the EOF pattern
    :param path, savename: Path to save directory and name to save the figure as
    '''

    # Set colorbar information 
    cmax = np.nanmax(np.abs(regression_patterns)); cmin = -1.0 * cmax
    cint = (cmax - cmin)/50
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'coolwarm', lut = nlevs)

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
    fig = plt.figure(figsize = [20,26])
    
    # Start with the regression pattern map
    ax1 = fig.add_subplot(3,2,(1,4), projection = proj) # 3 rows, 2 cols, and this figure will take the first two rows and two columns

    # Set the title
    ax1.set_title('%s Regression onto Mode %d Princple Component, Var Explained = %5.3f'%(variable.upper(), mode, var_explained), fontsize = 22)

    # Add ocean features
    ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Add coastlines and country borders
    ax1.coastlines(edgecolor = 'black', zorder = 3)
    ax1.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

	# Set the tick information
    ax1.set_yticklabels(LatLabel, fontsize = 26)
    ax1.yaxis.set_major_formatter(LatFormatter)
    
    ax1.set_xticklabels(LonLabel, fontsize = 26)
    ax1.xaxis.set_major_formatter(LonFormatter)
    
    # Adjust the ticks
    ax1.set_xticks(LonLabel, crs = proj)
    ax1.set_yticks(LatLabel, crs = proj)

    # Plot the EOF regression patterns
    cs = ax1.pcolormesh(lon, lat, regression_patterns, vmin = cmin, vmax = cmax,
                        cmap = cmap, transform = proj, zorder = 1)
        
    # Set the colorbar size and location
    cbax = fig.add_axes([0.840, 0.3885, 0.030, 0.480])

    label = 'Regression Slope'

    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = 'both', orientation = 'vertical')

    # Make the colorbar label
    cbar.ax.set_ylabel(label, fontsize = 22)

    # Set the colorbar tick size
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(22)

    # Set the map extent
    ax1.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


	# Make the PC time series plot
    ax2 = fig.add_subplot(3,2,(5,6))

    # Set the title
    ax2.set_title('Principle Component Timeseries for Mode %d'%mode, fontsize = 22)

    # Plot the PC
    ax2.plot(times, pcs, 'b-', linewidth = '2')
    # Add a 0 line for reference
    ax2.axhline(0, 0, 1, linestyle = '--', color = 'k', linewidth = 2)

    # Set the labels
    ax2.set_xlabel('Time', fontsize = 22)
    ax2.set_ylabel('Index - %s'%variable.upper(), fontsize = 22)

    # Set ticks formats
    years = YearLocator(5)
    months = MonthLocator(10)

    years_format = DateFormatter('%Y')

    ax2.xaxis.set_major_locator(years)
    ax2.xaxis.set_minor_locator(months)
    ax2.xaxis.set_major_formatter(years_format)

	# Set the tick size
    for i in ax2.xaxis.get_ticklabels() + ax2.yaxis.get_ticklabels():
        i.set_size(22)


    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


def create_regional_boxes(
        lc, 
        lat, 
        lon, 
        legend_labels, 
        path: str = './',
        savename: str = 'tmp.png', 
        ) -> None:
    '''
    Create a map displaying land cover and how the map is split into different subregions

    Inputs:
    :param lc: Land cover data to plot
    :param lat, lon: Latitudes and longitudes for the land cover
    :param legend_labels: List of labels for all possible land cover types for the legend
    :param path, savename: Path to save directory and name to save the figure as
    '''

    # Set the borders of the subregions
    borders = [ # Order is [min_lat, max_lat, min_lon, max_lon] 
        [4, 15, 340, 28], # (min and max lon refer to western 
        [-10, 4, 8, 28],  # and eastern lon respectively)
        [-10, 15, 28, 52],
        [-35, -10, 9, 41],
        [-26, -11, 43, 51],
    ]

    labels = [
        'SL', # Sahel
        'CB', # Congo Basin
        'EA', # Eastern Africa
        'SA', # Southern Africa
        'MD', # Madagascar
    ]

    colors = [ # Using colors from a Google Earth example (https://developers.google.com/earth-engine/guides/image_visualization#colab-python_8)
        '#aec3d4', # Water
        '#152106', # Evergreen Needleleaf
        '#225129', # Evergreen Broadleaf
        '#369b47', # Deciduous Needleleaf
        '#30eb5b', # Deciduous Broadleaf
        '#387242', # Mixed Forest
        '#6a2325', # Closed Shrubland
        '#c3aa69', # Open Shrublands
        '#b76031', # Woody Savannas
        '#d9903d', # Savannas
        '#91af40', # Grasslands
        '#111149', # Permanent Wetlands
        '#ff0000', # Croplands; Use the AI suggested color for better visibility
        # '#cdb33b', # Croplands
        '#cc0013', # Urban and Built Up
        '#33280d', # Cropland and Natural Vegetation Mosaic
        '#d7cdcc', # Snow and Ice
        '#f7e084', # Baren/Sparsely Vegetated
    ]

    # Determine which classes to exclude in the legend based on infrequent use (nearly invisible)
    classes = np.arange(0, 16+1)
    exclude_classes = np.array([np.nansum(lc == land_class) <= 100 for land_class in classes])
    exclude_indices = np.where(exclude_classes == True)[0]

    # Colorbar information
    cmin = 0; cmax = 16
    clevs = np.arange(cmin, cmax + 1)
    nlevs = len(clevs)
    cmap = mcolors.ListedColormap(colors)

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
    fig = plt.figure(figsize = [20,20])
    ax = fig.add_subplot(1,1,1, projection = proj)

    # Add coastlines and country borders
    ax.coastlines(edgecolor = 'black', zorder = 3)
    ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

	# Plot the land cover
    cs = ax.pcolormesh(lon, lat, lc, vmin = cmin, vmax = cmax, cmap = cmap, transform = proj, zorder = 1)

	# Set the tick formatting
    ax.set_yticklabels(LatLabel, fontsize = 26)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    ax.set_xticklabels(LonLabel, fontsize = 26)
    ax.xaxis.set_major_formatter(LonFormatter)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = proj)
    ax.set_yticks(LatLabel, crs = proj)
    
    # Add regional boxes
    for border, label in zip(borders, labels):
        # Determine borders and rectangular box information
        # Adjust to -180 to 180 longitude format for plotting mpatches.Rectangle
        height = border[1] - border[0]
        min_lat = border[0]
        if border[2] > 180:
            min_lon = border[2] - 360
        else:
            min_lon = border[2]

        if border[3] > 180:
            max_lon = border[3] - 360
        else:
            max_lon = border[3]
        width = max_lon - min_lon

        # Make the box for the region
        ax.add_patch(mpatches.Rectangle(xy = [min_lon, min_lat], 
                                        width = width, 
                                        height = height, 
                                        facecolor = 'none', 
                                        edgecolor = 'k', 
                                        linewidth = 4, 
                                        transform = proj, 
                                        zorder = 4))
        
        # Determine the label location
        if (min_lon > 180) & (max_lon < 180):
            central_lon = (min_lon - max_lon)/2
        else:
            central_lon = (min_lon + max_lon)/2
        # Add the label for the border
        ax.text(min_lon+1,#central_lon,           # Longitude (x) coordinate
                border[0]+1,#np.mean(border[:2]), # Latitude (y) coordinate
                label,
                color = 'k',
                bbox = dict(facecolor = 'white', edgecolor = None),
                fontsize = 26)

    # Set the map extent
    ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    # Make the patches for labels
    boxes = [mpatches.Rectangle((0,0), 2, 1, facecolor = color, edgecolor = 'k', label = label.title()) for color, label in zip(colors, legend_labels)]

    # Remove classes with few pixels
    for ind in reversed(exclude_indices):
        boxes.pop(ind)

    # Add boxes indicating significance colors
    fig.legend(handles = boxes, bbox_to_anchor = (1.28, 0.89), frameon = False, ncols = 1, fontsize = 26)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def create_single_map(
        data, 
        lat, 
        lon, 
        cbar_label, 
        path: str = './', 
        savename: str = 'tmp.png',
        ) -> None:
    '''
    Create a map displaying data

    Inputs:
    :param data: Two dimensional array of data to plot
    :param lat, lon: Latitudes and longitudes for the dataset
    :param cbar_label: Label for the colorbar
    :param path, savename: Path to save directory and name to save the figure as
    '''

    # Colorbar information
    cmin = 0; cmax = 2500 # np.ceil(np.nanmax(data))
    clevs = np.arange(cmin, cmax + 1)
    nlevs = len(clevs)
    cmap = plt.get_cmap(name = 'Greens', lut = nlevs)

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
    fig = plt.figure(figsize = [20,20])
    ax = fig.add_subplot(1,1,1, projection = proj)

    # Add ocean features
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

    # Add coastlines and country borders
    ax.coastlines(edgecolor = 'black', zorder = 3)
    ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

	# Plot the data
    cs = ax.pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax, cmap = cmap, transform = proj, zorder = 1)

	# Set the tick formatting
    ax.set_yticklabels(LatLabel, fontsize = 26)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    ax.set_xticklabels(LonLabel, fontsize = 26)
    ax.xaxis.set_major_formatter(LonFormatter)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = proj)
    ax.set_yticks(LatLabel, crs = proj)

    # Set the map extent
    ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    # Set the colorbar size and location
    cbax = fig.add_axes([0.880, 0.11, 0.030, 0.700])

    cbar = mcolorbar.Colorbar(cbax, mappable = cs, cmap = cmap, extend = 'max', orientation = 'vertical')

    # Make the colorbar label
    cbar.ax.set_ylabel(cbar_label, fontsize = 26)

    # Set the colorbar tick size
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(26)
   
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


if __name__ == '__main__':
    # Test and refine some of the figures
    from netCDF4 import Dataset

	# Determine which figure to test
    test_trend_sig = True
    test_correlation = False
    test_barplot = False
    test_scatterplot = False
    test_eof_plot = False

    # Determine FD types and path to the appropriate directory
    fd_types = ['sesr', 'rzsm', 'fdii']

    fn_bases = [
        'identified_fd/africa_fd_sesr_',
        'identified_fd/africa_fd_sm_%s_'%str(1),
        'fd_indices/africa_fdii_%s_'%str(1)
    ]

    var_list = ['tair', 'sp', 'd2m']

    # Load test dataset to obtain lats and lons
    with Dataset('%s/%s/%s2000.nc'%('../data', 'era5', fn_bases[0]), 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

	# Longitude correction for ERA5 data when plotting a subset that crosses 0 degree longitude
    lon_ind = np.where(lon[0,:] > 330)[0]
    lon_tmp = lon[:,lon_ind]
    lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)
    
    # Test the trend and significance maps
    if test_trend_sig:
        slope_map = []
        map_pval = []
        for fd_type in fd_types:
            # Generate random data
            slope = np.random.randn(lat.shape[0], lat.shape[1])
            slope_sum = np.random.randn(lat.shape[0], lat.shape[1])
            slope_win = np.random.randn(lat.shape[0], lat.shape[1])
            
            pval = np.random.random_sample((lat.shape))
            pval_sum = np.random.random_sample((lat.shape))
            pval_win = np.random.random_sample((lat.shape))

            slope_map.append([slope, slope_sum, slope_win])
            map_pval.append([pval, pval_sum, pval_win])

		# Make the trend maps for the slopes and slope significance
        make_trend_maps(
            slope_map,
            lat, 
            lon, 
            'characteristics', 
            fd_types, 
            ['Frequency', 'Duration', 'Normalized Severity'], 
            cmin = 0, 
            cmax = 3, 
            path = './', 
            savename = 'test_trend.png',
        )

        make_trend_maps(
            slope_map,
            lat, 
            lon, 
            'characteristics', 
            fd_types, 
            ['Frequency', 'Duration', 'Normalized Severity'], 
            sig = map_pval,
            cmin = 0, 
            cmax = 3, 
            significance_plots = True, 
            path = './', 
            savename = 'test_trend_sig.png',
        )

    # Test the correlation maps
    if test_correlation:
        r = {}; sig = {}
        r_lag = {}; sig_lag = {}

        lags = np.arange(-30, 30+1, 1)

        # Generate random data to plot for each FD type
        for fd_type in fd_types:

            for var in var_list:
                # Generate random data to fill in
                r['%s_%s'%(fd_type, var)] = np.random.random_sample((lat.shape))
                sig['%s_%s'%(fd_type, var)] = np.random.random_sample((lat.shape))

                r_lag['%s_%s'%(fd_type, var)] = np.random.random_sample((lags.size,))
                sig_lag['%s_%s'%(fd_type, var)] = np.random.random_sample((lags.size,))
            
        for var in var_list:
        	# Format the correlations and significances
            map_data = [r['%s_%s'%(fd_type, var)] for fd_type in fd_types]
            sig_data = [sig['%s_%s'%(fd_type, var)] for fd_type in fd_types]
            
            # Make the test maps
            savename = '%s_test_correlation_map.png'%(var)
            make_correlation_maps(
                map_data, 
                sig_data, 
                lat, 
                lon, 
                fd_types, 
                var, 
                path = './', 
                savename = savename,
            )

        # Make the test time series plots
        savename = 'test_lagged_correlation.png'
        make_lagged_correlation_plot(
            r_lag, 
            sig_lag, 
            lags, 
            var_list, 
            fd_types, 
            var_list, 
            path = './', 
            savename = savename,
        )

	# Test the bar plots
    if test_barplot:
        # For each fd type, generate some random data to plot
        spi_anomalies = {}
        pet_anomalies = {}
        for fd_type in fd_types:

            # Generate random data
            spi_anomalies[fd_type] = np.random.uniform(-3, 3, size = (80000,))
            pet_anomalies[fd_type] = np.random.uniform(-3, 3, size = (80000,))

        # Sort data into proper formatting
        bar_data = []
        bar_std = []
        for fd_type in fd_types:
            # Determine the occurrence of specific events (mimics how the barplot will be used)
            moisture_limited = np.where((spi_anomalies[fd_type] < -1) & (pet_anomalies[fd_type] < 1), 1, 0)
            energy_limited = np.where((spi_anomalies[fd_type] > -1) & (pet_anomalies[fd_type] > 1), 1, 0)
            both_limited = np.where((spi_anomalies[fd_type] < -1) & (pet_anomalies[fd_type] > 1), 1, 0)
            moisture_condition = np.where(spi_anomalies[fd_type] < -1, 1, 0)
            energy_condition = np.where(pet_anomalies[fd_type] > 1, 1, 0)

            # Relative percentage of conditions prior to FD events
            moisture_limited_regime = np.nansum(moisture_limited) * 100/spi_anomalies[fd_type].size
            energy_limited_regime = np.nansum(energy_limited) * 100/spi_anomalies[fd_type].size
            both_limited_regime = np.nansum(both_limited) * 100/spi_anomalies[fd_type].size
            moisture_condition_regime = np.nansum(moisture_condition) * 100/spi_anomalies[fd_type].size
            energy_condition_regime = np.nansum(energy_condition) * 100/spi_anomalies[fd_type].size

            bar_data.append([moisture_condition_regime, energy_condition_regime, moisture_limited_regime, energy_limited_regime, both_limited_regime])

            # Standard deviations of events
            moisture_limited_regime = np.nanstd(moisture_limited * 100/spi_anomalies[fd_type].size)*1700
            energy_limited_regime = np.nanstd(energy_limited * 100/spi_anomalies[fd_type].size)*1700
            both_limited_regime = np.nanstd(both_limited * 100/spi_anomalies[fd_type].size)*1700
            moisture_condition_regime = np.nanstd(moisture_condition * 100/spi_anomalies[fd_type].size)*1700
            energy_condition_regime = np.nanstd(energy_condition * 100/spi_anomalies[fd_type].size)*1700

            bar_std.append([moisture_condition_regime, energy_condition_regime, moisture_limited_regime, energy_limited_regime, both_limited_regime])

        # Tick labels
        x_ticks = ['SPI<-1', 'PET>1', 'SPI<-1 &\nPET<1', 'SPI>-1 &\nPET>1', 'SPI<-1 &\nPET>1']
        savename = 'test_barplot.png'
        # Make the test barplot
        make_barplots(
            bar_data, 
            fd_types, 
            'Relative Number (%) of FDs\nwith Given Conditions',
            x_ticks,
            bar_err = bar_std,
            path = './',
            savename = savename,
        )
        
    # Test the scatterplot
    if test_scatterplot:
        from scipy import stats
        from statistics_calculations import least_squares

        # Generate test data for FD counts, and r and pvals.
        freq = []
        freq_sum = []
        freq_win = []

        # Make the hypthesis testing for correlation
        rng = np.random.default_rng()
        test_method = stats.MonteCarloMethod(n_resamples = 100, rvs = (rng.normal, rng.normal))
        
        for fd_type in fd_types:
            # Generate data to plot
            if fd_type == 'sesr':
                initial_generation = np.random.randn(80000) * 20 + 50
                initial_generation_sum = np.random.randn(80000) * 20 + 50
                initial_generation_win = np.random.randn(80000) * 20 + 50
            else:
                initial_generation = initial_generation * 1.2 + (np.random.randn(80000)* 20 + 50)/4
                initial_generation_sum = initial_generation_sum * 0.8 + (np.random.randn(80000)* 20 + 50)/5
                initial_generation_win = initial_generation_win * 1.5 + (np.random.randn(80000)* 20 + 50)/3
            freq.append(initial_generation)
            freq_sum.append(initial_generation_sum)
            freq_win.append(initial_generation_win)

        # Calculate correlation and pval
        correlations = np.ones((len(freq), 3))
        pvals = np.ones((len(freq), 3))
        slopes = np.ones((len(freq), 3))
        intercepts = np.ones((len(freq), 3))
        for i in range(len(freq)):
            # Determine the other frequency to correlate the current one with
            alt_ind = i + 1
            if alt_ind >= len(freq):
                alt_ind = 0

            # Perform correlation calculations
            results = stats.pearsonr(freq[i], freq[alt_ind], method = test_method)
            correlations[i,0] = results.statistic
            pvals[i,0] = results.pvalue

            results = stats.pearsonr(freq_sum[i], freq_sum[alt_ind], method = test_method)
            correlations[i,1] = results.statistic
            pvals[i,1] = results.pvalue

            results = stats.pearsonr(freq_win[i], freq_win[alt_ind], method = test_method)
            correlations[i,2] = results.statistic
            pvals[i,2] = results.pvalue

            # Determine regression values
            slopes[i,0], intercepts[i,0], _ = least_squares(freq[i], freq[alt_ind])
            slopes[i,1], intercepts[i,1], _ = least_squares(freq_sum[i], freq_sum[alt_ind])
            slopes[i,2], intercepts[i,2], _ = least_squares(freq_win[i], freq_win[alt_ind])

		# Make the test scatterplots
        make_scatterplots(
            freq, 
            freq_sum, 
            freq_win, 
            correlations, 
            pvals, 
            fd_types, 
            ['Annual', 'MAMJJA', 'SONDJF'],
            slope = slopes, 
            intercept = intercepts,  
            path = './', 
            savename = 'test_scatterplots.png',
        )

	# Test the EOF plots
    if test_eof_plot:
        from datetime import datetime, timedelta

        # Create filler information
        start = datetime(1979, 1, 1)
        end = datetime(2024, 12, 31)
        Ndays = (end - start).days
        times = np.array([start + timedelta(days = day) for day in range(Ndays)])
        mode = 1
        var_explained = 50.423

        # Create dummy data
        dummy_patterns = np.random.randn(lat.shape[0], lat.shape[1])
        dummy_pc = np.random.randn(times.size)

        # Smooth the dummy data for the PC
        runmean = 90
        dummy_pc = np.convolve(dummy_pc, np.ones((runmean))/runmean)[(runmean-1):]
        
        # Make the test EOF plot
        make_eof_plot(
            dummy_patterns, 
            lat, 
            lon, 
            dummy_pc, 
            times, 
            'sesr', 
            mode, 
            var_explained, 
            path = './', 
            savename = 'test_eof.png',
        )


