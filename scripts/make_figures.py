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

# Domain for Africa
lower_lat = -35; upper_lat = 35
lower_lon = 335; upper_lon = 53
# lower_lat = -85; upper_lat = 85
# lower_lon = -180; upper_lon = 180

def make_statistics_maps(data, lat, lon, statistic, fd_types, times, cmin = 0, cmax = 50, path = './', savename = 'tmp.png'):
    """
    Make a set of maps for different FD statistics (frequency, intensity, or duration)
    """

    # Set colorbar information 
    cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
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
    y_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''}) # Middle
    x_formatter = cticker.LatitudeFormatter(cardinal_labels = {'west': '', 'east': ''}) # Middle

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
            # Set the title
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

            if (j == 0):
                axes[i,j].set_yticklabels(LatLabel, fontsize = 22)
                axes[i,j].yaxis.set_major_formatter(LatFormatter)
            else:
                axes[i,j].yaxis.set_major_formatter(y_formatter)
                axes[i,j].set_yticklabels('', fontsize = 0)

            if (i == 2):
                axes[i,j].set_xticklabels(LonLabel, fontsize = 22)
                axes[i,j].xaxis.set_major_formatter(LonFormatter)
            else:
                axes[i,j].xaxis.set_major_formatter(x_formatter)
                axes[i,j].set_xticklabels('', fontsize = 0)

            # Set y labels if necessary
            if j == 0:
                # Also set the label
                axes[i,j].set_ylabel(fd_types[i].upper(), fontsize = 22)
            
            # Plot the data
            cs = axes[i,j].pcolormesh(lon, lat, data[j][i], vmin = cmin, vmax = cmax,
                                      cmap = cmap, transform = proj, zorder = 1)
            
            # Set the map extent
            axes[i,j].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    # Set the colorbar size and location
    cbax = fig.add_axes([0.150, 0.0685, 0.72, 0.020])

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

def make_trend_maps(data, lat, lon, statistic, fd_types, times, cmin = -1, cmax = 1, significance_plots = False, path = './', savename = 'tmp.png'):
    """
    Make a set of maps for trends in FD characteristics (and statistical significance) (frequency, intensity, or duration)
    """

    alpha = 0.05
    units = {'frequency': 'Events / Grid Point / Year', 
             'duration': 'Days / Year', 
             'severity': 'Unitless / Year'}

    # Set colorbar information 
    cint = (cmax - cmin)/20
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    if significance_plots:
        cmap  = plt.get_cmap(name = 'Paired', lut = nlevs)
    else:
        if statistic == 'severity':
            cmap  = plt.get_cmap(name = 'BrBG', lut = nlevs)
        else:
            cmap  = plt.get_cmap(name = 'BrBG_r', lut = nlevs)

    # Lonitude and latitude tick information
    lat_int = 10
    lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    y_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''}) # Middle
    x_formatter = cticker.LatitudeFormatter(cardinal_labels = {'west': '', 'east': ''}) # Middle

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
            # Set the title
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

            if (j == 0):
                axes[i,j].set_yticklabels(LatLabel, fontsize = 22)
                axes[i,j].yaxis.set_major_formatter(LatFormatter)
            else:
                axes[i,j].yaxis.set_major_formatter(y_formatter)
                axes[i,j].set_yticklabels('', fontsize = 0)

            if (i == 2):
                axes[i,j].set_xticklabels(LonLabel, fontsize = 22)
                axes[i,j].xaxis.set_major_formatter(LonFormatter)
            else:
                axes[i,j].xaxis.set_major_formatter(x_formatter)
                axes[i,j].set_xticklabels('', fontsize = 0)

            # Set y labels if necessary
            if j == 0:
                # Also set the label
                axes[i,j].set_ylabel(fd_types[i].upper(), fontsize = 22)
            
            # Plot the data
            if significance_plots:
                # Determine areas of statistical significance
                significance_data = np.where((data[i][j] > (1-alpha/2)) | (data[i][j] < (alpha/2)), 1, 0)
                cs = axes[i,j].pcolormesh(lon, lat, significance_data, vmin = 0, vmax = 1,
                                          cmap = cmap, transform = proj, zorder = 1)
            else:
                cs = axes[i,j].pcolormesh(lon, lat, data[i][j], vmin = cmin, vmax = cmax,
                                          cmap = cmap, transform = proj, zorder = 1)
            
            # Set the map extent
            axes[i,j].set_extent([lower_lon, upper_lon, lower_lat, upper_lat])


    # Make the colorbar if needed
    if significance_plots == False:
        # Set the colorbar size and location
        cbax = fig.add_axes([0.150, 0.0685, 0.72, 0.020])

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
    
    else:
        # Custom patches
        not_sig = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap(0.0), edgecolor = 'k', label = 'Not Significant')
        sig = mpatches.Rectangle((0,0), 2, 1, facecolor = cmap(1.0), edgecolor = 'k', label = 'Significant')
        # Add boxes indicating significance colors
        fig.legend(handles = [not_sig, sig], bbox_to_anchor = (0.65, 0.1), frameon = False, ncols = len([not_sig, sig]), fontsize = 22) # loc = 'lower center',
            
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()

def make_correlation_maps(data, pvals, lat, lon, fd_types, var_name, index_corr = False, path = './', savename = 'tmp.png'):
    '''
    Make a map of correlations with statistical significances
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
    y_formatter = cticker.LatitudeFormatter(cardinal_labels = {'north': '', 'south': ''}) # Middle
    x_formatter = cticker.LatitudeFormatter(cardinal_labels = {'west': '', 'east': ''}) # Middle

    # Projection information
    proj = ccrs.PlateCarree()

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [16, 24], nrows = 2, ncols = 3, 
                             subplot_kw = {'projection': proj})
    plt.subplots_adjust(wspace = 0.12, hspace = -0.65)

    fig.suptitle('Correlation between FD and %s'%var_name, y = 0.72, fontsize = 22)
 
    # Loop through columns
    for j in range(3):
        # Set the title
        axes[0,j].set_title(fd_types[j].upper(), size = 22)

        # Add ocean features
        axes[0,j].add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)

        # Add coastlines and country borders
        axes[0,j].coastlines(edgecolor = 'black', zorder = 3)
        axes[0,j].add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        axes[0,j].set_xticks(LonLabel, crs = proj)
        axes[0,j].set_yticks(LatLabel, crs = proj)

        if (j == 0):
            axes[0,j].set_yticklabels(LatLabel, fontsize = 22)
            axes[0,j].yaxis.set_major_formatter(LatFormatter)
        else:
            axes[0,j].yaxis.set_major_formatter(y_formatter)
            axes[0,j].set_yticklabels('', fontsize = 0)

        axes[0,j].xaxis.set_major_formatter(x_formatter)
        axes[0,j].set_xticklabels('', fontsize = 0)
        # axes[0,j].set_xticklabels(LonLabel, fontsize = 22)
        # axes[0,j].xaxis.set_major_formatter(LonFormatter)

            # axes[j,0].xaxis.set_major_formatter(x_formatter)
            # axes[j,0].set_xticklabels('', fontsize = 0)
        
        # Plot the data
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

        if (j == 0):
            axes[1,j].set_yticklabels(LatLabel, fontsize = 22)
            axes[1,j].yaxis.set_major_formatter(LatFormatter)
        else:
            axes[1,j].yaxis.set_major_formatter(y_formatter)
            axes[1,j].set_yticklabels('', fontsize = 0)

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


def make_lagged_correlation_plot(data, pvals, lags, snames, fd_types, labels, path = './', savename = 'tmp.png'):
    '''
    Make a set of lagged correlation plots with significances
    '''
    alpha = 0.05
    ncols = 3
    # colors = ['k', 'r', 'b']

    # Initialize the figure
    fig, axes = plt.subplots(figsize = [28, 8], nrows = 1, ncols = ncols)

    for n in range(ncols):
        # Make the regression lines
        for m, sname in enumerate(snames):
            # Determine areas of statistical significance
            significance_data = [(pval > (1-alpha/2)) | (pval < (alpha/2)) for pval in pvals['%s_%s'%(fd_types[n], sname)]]
            #np.where((pvals['%s_%s'%(fd_types[n], sname)] > (1-alpha/2)) | (pvals[['%s_%s'%(fd_types[n], sname)] < (alpha/2)), 1, 0)

            # Plot the time series color = colors[m], 
            axes[n].plot(lags, data['%s_%s'%(fd_types[n], sname)], marker = 'd', markevery = significance_data, label = labels[m])

        # Set the plot title
        axes[n].set_title(fd_types[n].upper(), fontsize = 22)

        # Add the legend
        axes[n].legend(fontsize = 16)

        # Set the label
        axes[n].set_xlabel('Lagged Time [days]', fontsize = 22)
        if n == 0:
            axes[n].set_ylabel('Lagged Correlation Coefficient', fontsize = 22)

        # Set the tick size
        for i in axes[n].xaxis.get_ticklabels() + axes[n].yaxis.get_ticklabels():
            i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    return


def make_boxplots(data, fd_types, times, statistic, path = './', savename = 'tmp.png'):
    """
    Make a box and whiskers plot of the data
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
    if statistic == 'severity':
        plt.subplots_adjust(wspace = 0.45)

    # Perform plot for each column
    for j in range(3):

        # Set the title
        axes[j].set_title(fd_types[j].upper(), fontsize = 22)

        # Make the box and whiskers plot
        bplots = axes[j].boxplot(data[j], 
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
                                 showmeans = False)

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

def make_variable_boxplots(data, var_labels, labels, fd_type, path = './', savename = 'tmp.png'):
    '''
    Make a box plot for multiple variables during FD
    '''

    for n in range(len(data)):
        for m in range(len(data[n])):
            data[n][m] = np.delete(data[n][m], np.isnan(data[n][m]))

    # Initialize the figure
    fig, ax = plt.subplots(figsize = [14, 8], nrows = 1, ncols = 1)

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
        bplots = ax.boxplot(data[n], 
                            notch = True, 
                            whis = (5, 95),
                            positions = [n+1+5*m for m in range(len(data[n]))],
                            sym = '', 
                            vert = True, 
                            bootstrap = 1000, 
                            patch_artist = True, 
                            tick_labels = var_labels if n == 1 else ['' for _ in range(len(var_labels))],
                            manage_ticks = True, 
                            meanline = True,
                            showcaps = False,
                            showmeans = False)

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

def make_barplots(data, labels, ylabel, xtick_labels, bar_err = None, path = './', savename = 'tmp.png'):
    '''
    Make a boxplot of multiple inputs
    '''

    # Initialize the figure
    fig, ax = plt.subplots(figsize = [14, 8], nrows = 1, ncols = 1)

    colors = ['grey', 
              '#DC143C', # Crimson
              '#0000FF'  # Blue
              ]

    # Make the bar plots
    for n, dataset in enumerate(data):
        ind = np.arange(len(dataset))
        width = 0.25

        if bar_err is not None:
            err = bar_err[n]
        else:
            err = None
        print(err)

        ax.bar(ind + n*width, dataset, width, facecolor = colors[n], alpha = 1.0, yerr = err, label = labels[n].upper())

    # Set the legend
    ax.legend(fontsize = 22)

    # Set the ticks
    ax.set_xticks(ind + width, labels = xtick_labels)

    # Set t label
    ax.set_ylabel(ylabel, fontsize = 22)

    # Set the tick size
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


def timeseries_plot(x, y, slopes, intercepts, p_values, fd_types, label, times, path = './', savename = 'tmp.png'):
    '''
    Make multiple time series plots (columns of three), with multiple lines and regression lines for eachs
    '''

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
            yhat = slopes[m][n] * x + intercepts[m][n]

            # Plot the time series
            axes[n].plot(x, y[m][n], color = colors[m], marker = symbols[m], label = '')
            axes[n].plot(x, yhat, color = colors[m], linestyle = '--', marker = symbols[m], label = r'%s: $\hat{y}$ = %6.4ft+%5.2f, p-value = %4.3f'%(fd_types[m].upper(), slopes[m][n], intercepts[m][n], p_values[m][n]))

        # Set the plot title
        axes[n].set_title(times[n], fontsize = 22)

        # Add the legend
        axes[n].legend(fontsize = 16)

        # Set the label
        axes[n].set_xlabel('Time', fontsize = 22)
        if n == 0:
            axes[n].set_ylabel('Averaged %s [%s]'%(label.title(), units[label]), fontsize = 22)

        # Set the tick size
        for i in axes[n].xaxis.get_ticklabels() + axes[n].yaxis.get_ticklabels():
            i.set_size(22)

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

def make_scatterplots(x, y, z, r, pval, labels, suptitles, slope = None, intercept = None, path = './', savename = 'tmp.png'):
    '''
    Make a set of scatterplots between with potentially linear regression lines between variables
    '''
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

            h2d = ax.hist2d(data[i], data[alt_ind], bins = 60, cmin = cmin, vmax = cmax, edgecolor = 'face', cmap = cmap, rasterized = True)

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

def create_regional_boxes(savename = 'tmp.png', path = './'):
    '''
    Create a map displaying how an area is split into different regions
    '''

    # Set the borders
    borders = [ # Order is [min_lat, max_lat, min_lon, max_lon] 
        [4, 15, 340, 12], # (min and max lon refer to western 
        [-10, 4, 8, 35],  # and eastern lon respectively)
        [4, 15, 12, 35],
        [-10, 15, 35, 52],
        [-35, -10, 9, 41],
        [-26, -11, 43, 51],
    ]

    labels = [
        'Sahel',
        'Tropical Africa',
        'Central Africa',
        'Eastern Africa',
        'Southern Africa',
        'Madagascar',
    ]

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

    ax.set_yticklabels(LatLabel, fontsize = 26)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    ax.set_xticklabels(LonLabel, fontsize = 26)
    ax.xaxis.set_major_formatter(LonFormatter)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = proj)
    ax.set_yticks(LatLabel, crs = proj)
    
    # Add regional boxes
    for border, label in zip(borders, labels):
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

        # Add the border
        ax.add_patch(mpatches.Rectangle(xy = [min_lon, min_lat], 
                                        width = width, 
                                        height = height, 
                                        facecolor = 'none', 
                                        edgecolor = 'k', 
                                        linewidth = 3, 
                                        transform = proj, 
                                        zorder = 4))
        
        # Add the label for the border
        if (min_lon > 180) & (max_lon < 180):
            central_lon = (min_lon - max_lon)/2
        else:
            central_lon = (min_lon + max_lon)/2
        ax.text(min_lon+1,#central_lon,         # Longitude (x) coordinate
                border[0]+1,#np.mean(border[:2]), # Latitude (y) coordinate
                label,
                color = 'r',
                fontsize = 26)

    # Set the map extent
    ax.set_extent([lower_lon, upper_lon, lower_lat, upper_lat])

    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    plt.close()


if __name__ == '__main__':
    # Test and refine some of the figures
    from netCDF4 import Dataset

    test_correlation = False
    test_barplot = False
    test_scatterplot = False

    # Determine FD types
    fd_types = ['sesr', 'rzsm', 'fdii']

    fn_bases = [
        'identified_fd/africa_fd_sesr_',
        'identified_fd/africa_fd_sm_%s_'%str(1),
        'fd_indices/africa_fdii_%s_'%str(1)
    ]

    var_list = ['tair', 'sp', 'd2m']

    # Load test dataset to obtain lats and lons
    with Dataset('%s/%s2000.nc'%('../data', fn_bases[0]), 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

    lon_ind = np.where(lon[0,:] > 330)[0]
    lon_tmp = lon[:,lon_ind]
    lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)
    
    # Initialize some datasets
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
            map_data = [r['%s_%s'%(fd_type, var)] for fd_type in fd_types]
            sig_data = [sig['%s_%s'%(fd_type, var)] for fd_type in fd_types]
            
            # Make the test maps
            savename = '%s_test_correlation_map.png'%(var)
            make_correlation_maps(map_data, sig_data, lat, lon, fd_types, var, path = './', savename = savename)

        # Make the test time series plots
        savename = 'test_lagged_correlation.png'
        make_lagged_correlation_plot(r_lag, sig_lag, lags, var_list, fd_types, var_list, path = './', savename = savename)

    if test_barplot:
        # Test the barplot to ensure it comes out well.

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
            moisture_limited = np.where((spi_anomalies[fd_type] < -1) & (pet_anomalies[fd_type] < 1), 1, 0)
            energy_limited = np.where((spi_anomalies[fd_type] > -1) & (pet_anomalies[fd_type] > 1), 1, 0)
            both_limited = np.where((spi_anomalies[fd_type] < -1) & (pet_anomalies[fd_type] > 1), 1, 0)
            moisture_condition = np.where(spi_anomalies[fd_type] < -1, 1, 0)
            energy_condition = np.where(pet_anomalies[fd_type] > 1, 1, 0)

            # Relative percentage of FD events
            moisture_limited_regime = np.nansum(moisture_limited) * 100/spi_anomalies[fd_type].size
            energy_limited_regime = np.nansum(energy_limited) * 100/spi_anomalies[fd_type].size
            both_limited_regime = np.nansum(both_limited) * 100/spi_anomalies[fd_type].size
            moisture_condition_regime = np.nansum(moisture_condition) * 100/spi_anomalies[fd_type].size
            energy_condition_regime = np.nansum(energy_condition) * 100/spi_anomalies[fd_type].size

            bar_data.append([moisture_condition_regime, energy_condition_regime, moisture_limited_regime, energy_limited_regime, both_limited_regime])

            # Standard deviations of relative FD events
            moisture_limited_regime = np.nanstd(moisture_limited * 100/spi_anomalies[fd_type].size)*1700
            energy_limited_regime = np.nanstd(energy_limited * 100/spi_anomalies[fd_type].size)*1700
            both_limited_regime = np.nanstd(both_limited * 100/spi_anomalies[fd_type].size)*1700
            moisture_condition_regime = np.nanstd(moisture_condition * 100/spi_anomalies[fd_type].size)*1700
            energy_condition_regime = np.nanstd(energy_condition * 100/spi_anomalies[fd_type].size)*1700

            bar_std.append([moisture_condition_regime, energy_condition_regime, moisture_limited_regime, energy_limited_regime, both_limited_regime])

        # Tick labels
        x_ticks = ['SPI<-1', 'PET>1', 'SPI<-1 &\nPET<1', 'SPI>-1 &\nPET>1', 'SPI<-1 &\nPET>1']
        savename = 'test_barplot.png'
        make_barplots(bar_data, 
                      fd_types, 
                      'Relative Number (%) of FDs\nwith Given Conditions',
                      x_ticks,
                      bar_err = bar_std,
                      path = './',
                      savename = savename)
        
    if test_scatterplot:
        # Make a test run for scatter plots to refine them
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

            # Perform correlation
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

        make_scatterplots(freq, 
                          freq_sum, 
                          freq_win, 
                          correlations, 
                          pvals, 
                          fd_types, 
                          ['Annual', 'MAMJJA', 'SONDJF'],
                          slope = slopes, 
                          intercept = intercepts,  
                          path = './', 
                          savename = 'test_scatterplots.png')


