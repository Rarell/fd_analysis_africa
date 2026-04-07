import os, sys, warnings
import gc
import numpy as np
import multiprocessing as mp
from typing import Tuple
from datetime import datetime, timedelta
from scipy import stats
from netCDF4 import Dataset
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

from statistics_calculations import least_squares, correlate, monte_carlo_significance
from inputs_and_outputs import load_fd_one_year, load_index_one_year, load_raw_data, load_pickle, save_pickle
from make_figures import make_statistics_maps, make_boxplots, make_barplots, make_variable_boxplots, create_regional_boxes, make_trend_maps, timeseries_plot, make_correlation_maps, make_lagged_correlation_plot, make_scatterplots
from utils import standardize_variable, calculate_spi

warnings.filterwarnings('ignore')

def calculate_fd_characteristics(timeseries, index_ts, dates, mask, thresholds, j, fdii = False):
    '''
    Calculate the frequency and duration
    '''
    # Make the months and days
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    # Construct dates for one year of data to index thresholds if needed
    if thresholds is not None:
        start_one_year = datetime(2012, 1, 1); end_one_year = datetime(2012, 12, 31)
        N_days_one_year = (end_one_year - start_one_year).days
        dates_one_year = np.array([start_one_year + timedelta(days = day) for day in range(N_days_one_year)])

        months_one_year = np.array([date.month for date in dates_one_year])
        days_one_year = np.array([date.day for date in dates_one_year])

    # T, I, J = fd.shape

    # Initialize duration
    duration = []
    severity = []
    duration_count = 0
    severity_count = 0

    if mask == 0:
        return np.nan, np.nan, np.nan, j
    
    # Examine the FD time series
    for t in range(timeseries.size):
        # If some time stamps are skipped, reset severity and duration counts
        if t >= 1:
            if ((dates[t] - dates[t-1]) > timedelta(days = 1)) & (duration_count > 0):
                duration.append(duration_count)
                severity.append(severity_count)
                duration_count = 0
                severity_count = 0
                
            # if duration_cout = 0 already, there is nothing to reset
                
        # If FD is occurring, increment the duration count
        if timeseries[t] > 0:
            # Increment duration
            duration_count = duration_count + 1

            # When calculating duration, also get the severity 
            if fdii:
                # For FDII, severity is just the accumulation of FDII values
                severity_count = severity_count + ((-1) * timeseries[t])
            else:
                # Determine severity by calculating the decifit from the 40th percentile
                if thresholds is not None:
                    ind = np.where( (dates[t].day == days_one_year) & (dates[t].month == months_one_year) )[0]
                    deficit_threshold = thresholds[ind]
                else:
                    ind = np.where( (dates[t].day == days) & (dates[t].month == months) )[0]
                    deficit_threshold = np.nanpercentile(index_ts[ind], 40)
                severity_count = severity_count + (index_ts[t] - deficit_threshold)

        # When the duration count > 0, and fd occurence is false, then FD ended
        elif (timeseries[t] == 0) & (duration_count > 0):
            # When FD ends, add the total count to the duration list and total severity to the seveirty list
            # (FD with the determined duration) & reset the duration and severity counts
            duration.append(duration_count)
            severity.append(severity_count)
            duration_count = 0
            severity_count = 0

        # When there is no FD, ensure the count is set to 0
        else:
            duration_count = 0

    # Count all durations (e.g., all FDs) to get the frequency
    frequency = len(duration)

    # Determine the average duration
    duration = np.nanmean(duration)
    # duration = duration if duration < 100 else np.nan # Deal with some strange and erroroneous 
                                                      # behavior of a few pixels in the Congo Basin
    
    # Determine the average severity
    severity = np.nanmean(severity)

    # Intensity is the change in index_ts percentiles from FD onset until FD starts

    return frequency, duration, severity, j

def determine_fd_starttimes(fd, dates, j):
    '''
    Determine the start times of FD in a given time series
    '''

    active_fd = False
    start_times = []
    start_times_ind = []
    for t in range(len(dates)):
        if (fd[t] > 0) & (active_fd == False):
            # Note FD has occurred
            active_fd = True
            start_times.append(dates[t])
            start_times_ind.append(t)

        # When FD is occurring, there is nothing to track
        elif (fd[t] > 0) & (active_fd == True):
            continue

        # When FD is 0, set the fd tracker to false
        else:
            active_fd = False

    return start_times, start_times_ind, j

def calculate_fd_statistics(args, fn_base, sname, index_base, ind_sname, mask, times = 'all', return_fd_and_indices = False):
    '''
    Calculate FD climatology statistics (frequency, severity, duration)
    '''

    # Find all the files for the identified FD
    fd_files = glob('%s/%s*.nc'%(args.data_path, fn_base), recursive = True)
    fd_files = np.sort(fd_files)
    # print(fd_files)

    # Find all index files for severity
    index_files = glob('%s/%s*.nc'%(args.data_path, index_base), recursive = True)
    index_files = np.sort(index_files)

    if ('rz' in fn_base) & (index_base is not None):
        index_files_1 = glob('/ourdisk/hpc/ai2es/sedris/fd_analysis/data/liquid_vsm/africa_volumetric_soil_water_layer_1_*.nc', recursive = True)
        index_files_2 = glob('/ourdisk/hpc/ai2es/sedris/fd_analysis/data/liquid_vsm/africa_volumetric_soil_water_layer_2_*.nc', recursive = True)
        index_files_1 = np.sort(index_files_1)
        index_files_2 = np.sort(index_files_2)

    # Load an FD file to initialize the frequency and duration datasets
    with Dataset(fd_files[0], 'r') as nc:
        fd = nc.variables[sname][:]

    # Initialize FD frequency data (total FDs in dataset; starts count at 0)
    _, I, J = fd.shape
    frequency = np.zeros((I, J))
    duration = np.zeros((I, J))
    severity = np.zeros((I, J))

    fd_total = []
    index_total = []
    time = []
    for t, file in tqdm(enumerate(fd_files), desc = 'Loading data'):
        # Load 1 year of FD
        fd, dates = load_fd_one_year(file, sname, times = times)

        # Get the time length for loaded FD data (will vary with leap years)
        T, _, _ = fd.shape

        # Add to the time set of FD and timestamp data
        # print(np.nansum(fd > 0))
        fd_total.append(fd)
        time.append(dates)

        # Load one year of data of the appropiate index
        if ind_sname is not None:
            if 'rz' in index_base:
                index_files_combined = [index_files_1[t], index_files_2[t]]
                ind_snames = ['swvl1', 'swvl2']
                index_data = load_index_one_year(index_files_combined, ind_snames, index_base, times = times, I = I, J = J)
            else:
                index_data = load_index_one_year(index_files[t], ind_sname, index_base, times = times, I = I, J = J)
        else:
            # Placeholder so index_total can still be called when using FDII without 
            # significant code changes or further bloating the params argument
            index_data = np.zeros((1, I, J)) * np.nan 

        # Add to the total set of index data
        # print(np.nansum(fd > 0))
        index_total.append(index_data)

    fd_total = np.concatenate(fd_total, axis = 0)
    time = np.concatenate(time)

    if ind_sname is not None:
        index_total = np.concatenate(index_total, axis = 0)
    else:
        # Placeholder so index_total can still be called when using FDII without 
        # significant code changes or further bloating the params argument
        index_total = np.zeros((1, I, J)) * np.nan 

    if return_fd_and_indices:
        if ind_sname is not None:
            return fd_total, index_total
        else: # FDII is the index
            return fd_total, fd_total

    # Calculate the duration and frequency
    for i in tqdm(range(I), desc = 'Calculating FD statistics'):
        if args.nprocesses > 1:
            # Collect the arguments needed for to load and process 1 year of data
            params = [(fd_total[:,i,j], index_total[:,i,j], time, mask[i,j], None, j, False if ind_sname is not None else True) for j in range(J)]

            # Use multiprocessing to load and process data; when joined, sum all the pieces
            with mp.Pool(args.nprocesses) as pool:
                data = pool.starmap(calculate_fd_characteristics, params) # Load and process data
                
                for datum in data:
                    freq = datum[0]; dur = datum[1]; sev = datum[2]; j = datum[3]

                    frequency[i,j] = freq
                    duration[i,j] = dur
                    severity[i,j] = sev

        else:
            for j in range(J):
                frequency[i,j], duration[i,j], severity[i,j], _ = calculate_fd_statistics(fd_total[:,i,j], 
                                                                                          index_total[:,i,j], 
                                                                                          time, 
                                                                                          mask[i,j],
                                                                                          None,
                                                                                          j,
                                                                                          fdii = False if ind_sname is not None else True,)

        # for t in range(1, T): # Start from t = 1; t = 0 is assumed to have no FD at the start of data
        #     FD_occurence = np.where( (fd[t,:,:] > 0) & np.invert(fd[t-1,:,:] > 0), 1, 0)
        #     # Find all points where FD was recorded, but where the previous time step had no FD
        #     # This should make it so multiple FD events are not counted multiple times 
        #     frequency = frequency + FD_occurence
    # print(np.nanmin(frequency), np.nanmax(frequency), np.nanmean(frequency))
    # print(np.nanmin(duration), np.nanmax(duration), np.nanmean(duration))
    frequency[frequency == 0] = np.nan
    duration[duration == 0] = np.nan
    severity[severity == 0] = np.nan
        
    return frequency, duration, severity


def calculate_fd_statistics_by_year(args, fn_base, sname, index_base, ind_sname, mask, fd_type = None, times = 'all'):
    '''
    Calculate FD climatology statistics (frequency, severity, duration) for each year
    '''

    # Determine if the calculations has already been done
    level = 'rz' if args.level == 0 else str(args.level)
    filename = '%s_fd_characteristics_by_year_%s_level_%s.pkl'%(fd_type, times, level)
    if os.path.exists('%s/%s'%(args.data_path, filename)):
        frequency, duration, severity = load_pickle('%s/%s'%(args.data_path, filename))
        return frequency, duration, severity

    # Find all the files for the identified FD
    # Note files are data for 1 year of data
    fd_files = glob('%s/%s*.nc'%(args.data_path, fn_base), recursive = True)
    fd_files = np.sort(fd_files)
    # print(fd_files)

    # Find all index files for severity
    index_files = glob('%s/%s*.nc'%(args.data_path, index_base), recursive = True)
    index_files = np.sort(index_files)

    if ('rz' in fn_base) & (index_base is not None):
        index_files_1 = glob('/ourdisk/hpc/ai2es/sedris/fd_analysis/data/liquid_vsm/africa_volumetric_soil_water_layer_1_*.nc', recursive = True)
        index_files_2 = glob('/ourdisk/hpc/ai2es/sedris/fd_analysis/data/liquid_vsm/africa_volumetric_soil_water_layer_2_*.nc', recursive = True)
        index_files_1 = np.sort(index_files_1)
        index_files_2 = np.sort(index_files_2)

    # Load an FD file to initialize the frequency and duration datasets
    with Dataset(fd_files[0], 'r') as nc:
        fd = nc.variables[sname][:]

    # Load 40th percentile threshold (for severity calculations) if necessary
    if ind_sname is not None:
        with Dataset('%s/africa_%s_40_percent_thresh.nc'%(args.data_path, ind_sname), 'r') as nc:
            thresholds = nc.variables['thresholds'][:]

    # Initialize FD frequency data (total FDs in dataset; starts count at 0)
    _, I, J = fd.shape
    T = len(fd_files)
    frequency = np.zeros((T, I, J))
    duration = np.zeros((T, I, J))
    severity = np.zeros((T, I, J))

    time = []
    for t, file in tqdm(enumerate(fd_files), desc = 'Loading data and calculating statistics'):
        # Load 1 year of FD
        fd, dates = load_fd_one_year(file, sname, times = times)
        
        T, _, _ = fd.shape
        time.append(dates)

        # Load one year of data of the appropiate index
        if ind_sname is not None:
            if 'rz' in index_base:
                index_files_combined = [index_files_1[t], index_files_2[t]]
                ind_snames = ['swvl1', 'swvl2']
                index_data = load_index_one_year(index_files_combined, ind_snames, index_base, times = times, I = I, J = J)
            else:
                index_data = load_index_one_year(index_files[t], ind_sname, index_base, times = times, I = I, J = J)
        else:
            # Placeholder so index_total can still be called when using FDII without 
            # significant code changes or further bloating the params argument
            index_data = np.zeros((1, I, J)) * np.nan 
            thresholds = np.zeros((1, I, J)) * np.nan 

        for i in range(I):
            if args.nprocesses > 1:
                # Collect the arguments needed for to load and process 1 year of data
                params = [(fd[:,i,j], index_data[:,i,j], dates, mask[i,j], thresholds[:,i,j], j, False if ind_sname is not None else True) for j in range(J)]

                # Use multiprocessing to load and process data; when joined, sum all the pieces
                with mp.Pool(args.nprocesses) as pool:
                    data = pool.starmap(calculate_fd_characteristics, params) # Load and process data
                    
                    for datum in data:
                        freq = datum[0]; dur = datum[1]; sev = datum[2]; j = datum[3]

                        frequency[t,i,j] = freq
                        duration[t,i,j] = dur
                        severity[t,i,j] = sev

            else:
                for j in range(J):
                    frequency[t,i,j], duration[t,i,j], severity[t,i,j], _ = calculate_fd_statistics(fd[:,i,j], 
                                                                                                    index_data[:,i,j], 
                                                                                                    dates, 
                                                                                                    mask[i,j],
                                                                                                    thresholds[:,i,j],
                                                                                                    j,
                                                                                                    fdii = False if ind_sname is not None else True,)

        # for t in range(1, T): # Start from t = 1; t = 0 is assumed to have no FD at the start of data
        #     FD_occurence = np.where( (fd[t,:,:] > 0) & np.invert(fd[t-1,:,:] > 0), 1, 0)
        #     # Find all points where FD was recorded, but where the previous time step had no FD
        #     # This should make it so multiple FD events are not counted multiple times 
        #     frequency = frequency + FD_occurence
    # print(np.nanmin(frequency), np.nanmax(frequency), np.nanmean(frequency))
    # print(np.nanmin(duration), np.nanmax(duration), np.nanmean(duration))
    time = np.concatenate(time)
    frequency[frequency == 0] = np.nan
    duration[duration == 0] = np.nan
    severity[severity == 0] = np.nan

    # Save results to pickle file so the calculations don't need to be repeated
    save_pickle('%s/%s'%(args.data_path, filename), [frequency, duration, severity], ['freq', 'dur', 'sev'])
        
    return frequency, duration, severity

def load_start_times(args, fn_base, sname):
    '''
    Determine the start dates of flash drought for a given method
    '''

    # Find all the files for the identified FD
    fd_files = glob('%s/%s*.nc'%(args.data_path, fn_base), recursive = True)
    fd_files = np.sort(fd_files)

    fd_total = []
    time = []

    for n, file in tqdm(enumerate(fd_files), desc = 'Loading data'):
        # Load 1 year of FD
        fd, dates = load_fd_one_year(file, sname, times = 'all')

        # Add to the time set of FD and timestamp data
        # print(np.nansum(fd > 0))
        fd_total.append(fd)
        time.append(dates)

    fd_total = np.concatenate(fd_total, axis = 0)
    time = np.concatenate(time)
    
    T, I, J = fd_total.shape
    fd_total = fd_total.reshape(T, I*J)

    fd_start_times = {}
    fd_start_times_ind = {}

    # Calculate the duration and frequency

    if args.nprocesses > 1:
        # Collect the arguments needed for to load and process 1 year of data
        params = [(fd_total[:,ij], time, ij) for ij in range(I*J)]

        # Use multiprocessing to load and process data; when joined, sum all the pieces
        with mp.Pool(args.nprocesses) as pool:
            data = pool.starmap(determine_fd_starttimes, params) # Load and process data
            
            for datum in data:
                start_times = datum[0]; start_times_ind = datum[1]; ij = datum[2]

                fd_start_times[ij] = start_times
                fd_start_times_ind[ij] = start_times_ind

    else:
        for ij in tqdm(range(I*J), desc = 'Calculating FD start times'):
            start_times, start_times_ind, _ = determine_fd_starttimes(fd_total[:,ij], time, ij)
            fd_start_times[ij] = start_times
            fd_start_times_ind[ij] = start_times_ind

    return fd_start_times, fd_start_times_ind

def trend_analysis(x, y):
    '''
    Perform a linear regression and significance test between two variables (assumed x is 1D)
    '''

    # Reshape y to 2D if needed
    if len(y.shape) > 1:
        T, I, J = y.shape
        y = y.reshape(T, I*J)

    # Perform the linear regression
    slope, intercept, yhat = least_squares(x, y)

    # Perform significance test (via Monte-Carlo Bootstrapping)
    pval = monte_carlo_significance(x.copy(), y.copy(), slope.copy())

    # Reshape if necessary
    if len(y.shape) > 1:
        slope = slope.reshape(I, J)
        pval = pval.reshape(I, J)

    return slope, intercept, pval
    

if __name__ == '__main__':
    # Create a parser to parse and process command line inputs
    description = 'Analysis ERA5 FD results over Africa'
    parser = ArgumentParser(description=description, fromfile_prefix_chars='@')

    # Pathing information
    parser.add_argument('--data_path', type = str, default = '/ourdisk/hpc/ai2es/sedris/fd_analysis/data', help = 'Path to FD datasets')
    parser.add_argument('--figure_path', type = str, default = '/ourdisk/hpc/ai2es/sedris/fd_analysis/figures', help = 'path to where the created figures are stored')

    # Type of analysis to perform
    parser.add_argument('--fd_stats_analysis', action = 'store_true', help = 'Make plots of FD statistics (frequency, seveirty, and duration)')
    parser.add_argument('--fd_trend_analysis', action = 'store_true', help = 'Make plots of FD trends (of characteristics) and their statistical signifcance')
    parser.add_argument('--fd_sensitivity_analysis', action = 'store_true', help = 'Make plots to examine the sensitivity of FD characteristics to different variables and determine the driving variables of FD')
    parser.add_argument('--fd_scatterplots', action = 'store_true', help = 'Make scatter plots of FD/indices compared with each other')
    parser.add_argument('--make_region_map', action = 'store_true', help = 'Make a map showing the different regions investigated')

    parser.add_argument('--skip_variable_boxplots', action = 'store_false', help = 'Skip making the variable anomaly boxplots in the sensitivity analysis')
    parser.add_argument('--skip_correlation_plots', action = 'store_false', help = 'Skip the correlation analysis in the sensitivity analysis')
    parser.add_argument('--skip_energy_moisture_drivers', action = 'store_false', help = 'Skip the moisture/energy limited analysis')

    parser.add_argument('--level', type = int, default = 0, help = 'ERA5 soil moisture level (must be 0 - 4; 0 means root zone depth)')
    parser.add_argument('--start_year', type = int, default = 1979, help = 'First year in FD dataset')
    parser.add_argument('--end_year', type = int, default = 2024, help = 'Last year in FD dataset')
    parser.add_argument('--nprocesses', type=int, default=1, help='Number of working threads for multiprocesses tasks')

    # Parse the arguments
    args = parser.parse_args()

    # Define the soil moisture level:
    level = 'rz' if args.level == 0 else str(args.level)

    # Types of FD analysis; sesr = Christian et al. 2023 method 
    #                       rzsm = Yuan et al. 2019 method 
    #                       fdii = Otkin et al. 2021 method
    fd_types = ['sesr', 'rzsm', 'fdii']

    # Define the .nc keys for the FD datasets
    fd_snames = ['fd', 'fd%s'%level, 'dro_sev%s'%level]
    index_snames = ['sesr', 'swvl%s'%level, None]

    # Define the base of the filename for the FD datasets
    fn_bases = [
        'identified_fd/africa_fd_sesr_',
        'identified_fd/africa_fd_sm_%s_'%level,
        'fd_indices/africa_fdii_%s_'%level
    ]

    index_bases = [
        'fd_indices/africa_sesr_',
        'liquid_vsm/africa_volumetric_soil_water_layer_%s_'%level,
        None
    ]

    # Load test dataset to obtain lats and lons
    with Dataset('%s/%s2000.nc'%(args.data_path, fn_bases[0]), 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

    # Load the mask
    with Dataset('%s/aridity_mask.nc'%args.data_path, 'r') as nc:
        mask = nc.variables['aim'][0,:,:]

    lon_ind = np.where(lon[0,:] > 330)[0]
    lon_tmp = lon[:,lon_ind]
    lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)

    # Make figures for FD statistics if desired
    if args.fd_stats_analysis:
        # Perform calculations for each type of FD identified
        frequency = []
        severity = []
        duration = []

        frequency_sum = []
        severity_sum = []
        duration_sum = []

        frequency_win = []
        severity_win = []
        duration_win = []

        for fd_type, sname, fn_base, ind_sname, ind_base in zip(fd_types, fd_snames, fn_bases, index_snames, index_bases):
            print(fd_type)

            # Perform FD statistics calculations for the full year
            freq, dur, sev = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask)

            tmp = freq[:,lon_ind]
            freq = np.concatenate([tmp, freq[:,:lon_ind[0]]], axis = 1)

            tmp = dur[:,lon_ind]
            dur = np.concatenate([tmp, dur[:,:lon_ind[0]]], axis = 1)

            tmp = sev[:,lon_ind]
            sev = np.concatenate([tmp, sev[:,:lon_ind[0]]], axis = 1)

            frequency.append(freq)
            duration.append(dur)
            severity.append(sev)

            # Perform FD statistics calculations for the "summer"
            freq, dur, sev = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask , times = 'summer')

            tmp = freq[:,lon_ind]
            freq = np.concatenate([tmp, freq[:,:lon_ind[0]]], axis = 1)

            tmp = dur[:,lon_ind]
            dur = np.concatenate([tmp, dur[:,:lon_ind[0]]], axis = 1)

            tmp = sev[:,lon_ind]
            sev = np.concatenate([tmp, sev[:,:lon_ind[0]]], axis = 1)

            frequency_sum.append(freq)
            duration_sum.append(dur)
            severity_sum.append(sev)

            # Perform FD statistics calculations for the "winter"
            freq, dur, sev = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask, times = 'winter')

            tmp = freq[:,lon_ind]
            freq = np.concatenate([tmp, freq[:,:lon_ind[0]]], axis = 1)

            tmp = dur[:,lon_ind]
            dur = np.concatenate([tmp, dur[:,:lon_ind[0]]], axis = 1)

            tmp = sev[:,lon_ind]
            sev = np.concatenate([tmp, sev[:,:lon_ind[0]]], axis = 1)

            frequency_win.append(freq)
            duration_win.append(dur)
            severity_win.append(sev)


        # Make maps of statistics
        frequencies = [frequency, frequency_sum, frequency_win]
        savename = 'frequency_maps_%s.png'%level
        make_statistics_maps(frequencies, lat, lon, 'frequency', fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = 0, cmax = 50, path = args.figure_path, savename = savename)

        durations = [duration, duration_sum, duration_win]
        savename = 'duration_maps_%s.png'%level
        make_statistics_maps(durations, lat, lon, 'duration', fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = 0, cmax = 50, path = args.figure_path, savename = savename)

        # For comparative analysis with different indices, normalize the severities
        max_sev = [np.nanmax(np.abs(severity[n])) for n in range(len(fd_types))]
        severity = [severity[n]/max_sev[n] for n in range(len(fd_types))]
        severity_sum = [severity_sum[n]/max_sev[n] for n in range(len(fd_types))]
        severity_win = [severity_win[n]/max_sev[n] for n in range(len(fd_types))]
        severities = [severity, severity_sum, severity_win]
        savename = 'severity_maps_%s.png'%level
        make_statistics_maps(severities, lat, lon, 'severity', fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = -1, cmax = 0, path = args.figure_path, savename = savename)
        
        # Make box plots of statistics
        box_data = [[frequency[n].flatten(), frequency_sum[n].flatten(), frequency_win[n].flatten()] for n in range(len(fd_types))]
        savename = 'frequency_boxplots_%s.png'%level
        make_boxplots(box_data, fd_types, ['Annual', 'MAMJJA', 'SONDJF'], 'frequency', path = args.figure_path, savename = savename)

        box_data = [[duration[n].flatten(), duration_sum[n].flatten(), duration_win[n].flatten()] for n in range(len(fd_types))]
        savename = 'duration_boxplots_%s.png'%level
        make_boxplots(box_data, fd_types, ['Annual', 'MAMJJA', 'SONDJF'], 'duration', path = args.figure_path, savename = savename)

        box_data = [[severity[n].flatten(), severity_sum[n].flatten(), severity_win[n].flatten()] for n in range(len(fd_types))]
        savename = 'severity_boxplots_%s.png'%level
        make_boxplots(box_data, fd_types, ['Annual', 'MAMJJA', 'SONDJF'], 'severity', path = args.figure_path, savename = savename)

    # Trends analysis:
        # Load in one year, do characteristic calculations, load next year, repeat to get annual average per year
        # Then do trend (plot maps and plot time series with regression line)
        # Look at trends over a specific region
    if args.fd_trend_analysis:
        slopes = {}; pvals = {}
        slopes['frequency'] = []; slopes['duration'] = []; slopes['severity'] = []
        pvals['frequency'] = []; pvals['duration'] = []; pvals['severity'] = []

        overall_slopes = []; overall_slopes_pval = []
        overall_intercepts = []
        overall_ts = []

        # Perform calculations for each type of FD identified for each year
        years = np.arange(1979, 2024+1)
        for fd_type, sname, fn_base, ind_sname, ind_base in zip(fd_types, fd_snames, fn_bases, index_snames, index_bases):
            print(fd_type)

            # Perform FD statistics calculations for the full year
            frequency, duration, severity = calculate_fd_statistics_by_year(args, fn_base, sname, ind_base, ind_sname, mask, fd_type = fd_type)
            tmp = frequency[:,:,lon_ind]
            frequency = np.concatenate([tmp, frequency[:,:,:lon_ind[0]]], axis = 2)

            tmp = duration[:,:,lon_ind]
            duration = np.concatenate([tmp, duration[:,:,:lon_ind[0]]], axis = 2)

            tmp = severity[:,:,lon_ind]
            severity = np.concatenate([tmp, severity[:,:,:lon_ind[0]]], axis = 2)

            # Perform FD statistics calculations for the "summer"
            frequency_sum, duration_sum, severity_sum = calculate_fd_statistics_by_year(args, fn_base, sname, ind_base, ind_sname, mask, fd_type = fd_type, times = 'summer')
            tmp = frequency_sum[:,:,lon_ind]
            frequency_sum = np.concatenate([tmp, frequency_sum[:,:,:lon_ind[0]]], axis = 2)

            tmp = duration_sum[:,:,lon_ind]
            duration_sum = np.concatenate([tmp, duration_sum[:,:,:lon_ind[0]]], axis = 2)

            tmp = severity_sum[:,:,lon_ind]
            severity_sum = np.concatenate([tmp, severity_sum[:,:,:lon_ind[0]]], axis = 2)

            # Perform FD statistics calculations for the "winter"
            frequency_win, duration_win, severity_win = calculate_fd_statistics_by_year(args, fn_base, sname, ind_base, ind_sname, mask, fd_type = fd_type, times = 'winter')
            tmp = frequency_win[:,:,lon_ind]
            frequency_win = np.concatenate([tmp, frequency_win[:,:,:lon_ind[0]]], axis = 2)

            tmp = duration_win[:,:,lon_ind]
            duration_win = np.concatenate([tmp, duration_win[:,:,:lon_ind[0]]], axis = 2)

            tmp = severity_win[:,:,lon_ind]
            severity_win = np.concatenate([tmp, severity_win[:,:,:lon_ind[0]]], axis = 2)

            # Normalize the severity so the slope can be uniformly be displayed on maps and plots
            max_sev = np.nanmax(np.abs(severity))
            severity = severity/max_sev; severity_sum = severity_sum/max_sev; severity_win = severity_win/max_sev

            T, I, J = frequency.shape
            # Perform trend analysis
            slope_freq, _, pval_freq = trend_analysis(years.copy(), frequency.copy())
            slope_dur, _, pval_dur = trend_analysis(years.copy(), duration.copy())
            slope_sev, _, pval_sev = trend_analysis(years.copy(), severity.copy())

            T, I, J = frequency_sum.shape
            # Repeat for "summer"
            slope_freq_sum, _, pval_freq_sum = trend_analysis(years.copy(), frequency_sum.copy())
            slope_dur_sum, _, pval_dur_sum = trend_analysis(years.copy(), duration_sum.copy())
            slope_sev_sum, _, pval_sev_sum = trend_analysis(years.copy(), severity_sum.copy())

            T, I, J = frequency_win.shape
            # Repeat for "winter"
            slope_freq_win, _, pval_freq_win = trend_analysis(years.copy(), frequency_win.copy())
            slope_dur_win, _, pval_dur_win = trend_analysis(years.copy(), duration_win.copy())
            slope_sev_win, _, pval_sev_win = trend_analysis(years.copy(), severity_win.copy())

            # Add the trend variables to their respective lists
            slope_freq[slope_freq == 0] = np.nan; pval_freq[np.isnan(slope_freq)] = np.nan
            slope_sev[slope_sev == 0] = np.nan; pval_sev[np.isnan(slope_sev)] = np.nan
            slope_dur[slope_dur == 0] = np.nan; pval_dur[np.isnan(slope_dur)] = np.nan

            slope_freq_sum[slope_freq_sum == 0] = np.nan; pval_freq_sum[np.isnan(slope_freq_sum)] = np.nan
            slope_sev_sum[slope_sev_sum == 0] = np.nan; pval_sev_sum[np.isnan(slope_sev_sum)] = np.nan
            slope_dur_sum[slope_dur_sum == 0] = np.nan; pval_dur_sum[np.isnan(slope_dur_sum)] = np.nan

            slope_freq_win[slope_freq_win == 0] = np.nan; pval_freq_win[np.isnan(slope_freq_win)] = np.nan
            slope_sev_win[slope_sev_win == 0] = np.nan; pval_sev_win[np.isnan(slope_sev_win)] = np.nan
            slope_dur_win[slope_dur_win == 0] = np.nan; pval_dur_win[np.isnan(slope_dur_win)] = np.nan

            slopes['frequency'].append([slope_freq, slope_freq_sum, slope_freq_win]) 
            pvals['frequency'].append([pval_freq, pval_freq_sum,pval_freq_win])

            slopes['duration'].append([slope_dur, slope_dur_sum, slope_dur_win]) 
            pvals['duration'].append([pval_dur, pval_dur_sum,pval_dur_win])

            slopes['severity'].append([slope_sev, slope_sev_sum, slope_sev_win]) 
            pvals['severity'].append([pval_sev, pval_sev_sum,pval_sev_win])

            # Peform trend analysis of overall means
            # tmp_freq = np.nanmean(frequency, axis = -1)
            tmp_freq = np.nanmean(frequency.reshape(T,I*J), axis = -1)
            tmp_dur = np.nanmean(duration.reshape(T,I*J), axis = -1)
            tmp_sev = np.nanmean(severity.reshape(T,I*J), axis = -1)
            slope_freq, intercept_freq, pval_freq = trend_analysis(years.copy(), tmp_freq.copy())
            slope_dur, intercept_dur, pval_dur = trend_analysis(years.copy(), tmp_dur.copy())
            slope_sev, intercept_sev, pval_sev = trend_analysis(years.copy(), tmp_sev.copy())

            # Repeat for "summer"
            tmp_freq_sum = np.nanmean(frequency_sum.reshape(T,I*J), axis = -1)
            tmp_dur_sum = np.nanmean(duration_sum.reshape(T,I*J), axis = -1)
            tmp_sev_sum = np.nanmean(severity_sum.reshape(T,I*J), axis = -1)
            slope_freq_sum, intercept_freq_sum, pval_freq_sum = trend_analysis(years.copy(), tmp_freq_sum.copy())
            slope_dur_sum, intercept_dur_sum, pval_dur_sum = trend_analysis(years.copy(), tmp_dur_sum.copy())
            slope_sev_sum, intercept_sev_sum, pval_sev_sum = trend_analysis(years.copy(), tmp_sev_sum.copy())

            # Repeat for "winter"
            tmp_freq_win = np.nanmean(frequency_win.reshape(T,I*J), axis = -1)
            tmp_dur_win = np.nanmean(duration_win.reshape(T,I*J), axis = -1)
            tmp_sev_win = np.nanmean(severity_win.reshape(T,I*J), axis = -1)
            slope_freq_win, intercept_freq_win, pval_freq_win = trend_analysis(years.copy(), tmp_freq_win.copy())
            slope_dur_win, intercept_dur_win, pval_dur_win = trend_analysis(years.copy(), tmp_dur_win.copy())
            slope_sev_win, intercept_sev_win, pval_sev_win = trend_analysis(years.copy(), tmp_sev_win.copy())

            # Append the spatial averages to their respective lists
            overall_slopes.append([slope_freq, slope_freq_sum, slope_freq_win]); overall_slopes_pval.append([pval_freq, pval_freq_sum, pval_freq_win])
            overall_slopes.append([slope_dur, slope_dur_sum, slope_dur_win]); overall_slopes_pval.append([pval_dur, pval_dur_sum, pval_dur_win])
            overall_slopes.append([slope_sev, slope_sev_sum, slope_sev_win]); overall_slopes_pval.append([pval_sev, pval_sev_sum, pval_sev_win])

            overall_intercepts.append([intercept_freq, intercept_freq_sum, intercept_freq_win])
            overall_intercepts.append([intercept_dur, intercept_dur_sum, intercept_dur_win])
            overall_intercepts.append([intercept_sev, intercept_sev_sum, intercept_sev_win])

            overall_ts.append([tmp_freq, tmp_freq_sum, tmp_freq_win])
            overall_ts.append([tmp_dur, tmp_dur_sum, tmp_dur_win])
            overall_ts.append([tmp_sev, tmp_sev_sum, tmp_sev_win])
            

        # Make maps
        characteristic = ['frequency', 'duration', 'severity']
        labels = ['FD Frequency', 'FD Duration', 'FD Severity']
        cmins = [-0.06, -0.5, -0.0015]
        cmaxes = [0.06, 0.5, 0.0015]

        for n in range(len(characteristic)):
            # Make the maps
            map_data = slopes[characteristic[n]]
            savename = '%s_trends_maps_%s.png'%(characteristic[n], level)
            make_trend_maps(map_data, lat, lon, characteristic[n], fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = cmins[n], cmax = cmaxes[n], path = args.figure_path, savename = savename)

             # Plot the significance
            map_pval = pvals[characteristic[n]]
            savename = '%s_trends_significance_maps_%s.png'%(characteristic[n], level)
            make_trend_maps(map_pval, lat, lon, characteristic[n], fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = 0, cmax = 1, significance_plots = True, path = args.figure_path, savename = savename)

            # Make the time series plots
            savename = '%s_timeseries_trends_%s.png'%(characteristic[n], level)
            overall_ts_plot = [overall_ts[0+n], overall_ts[3+n], overall_ts[6+n]]
            overall_slopes_plot = [overall_slopes[0+n], overall_slopes[3+n], overall_slopes[6+n]]
            overall_intercepts_plot = [overall_intercepts[0+n], overall_intercepts[3+n], overall_intercepts[6+n]]
            overall_slopes_pval_plot = [overall_slopes_pval[0+n], overall_slopes_pval[3+n], overall_slopes_pval[6+n]]
            timeseries_plot(years, overall_ts_plot, overall_slopes_plot, overall_intercepts_plot, overall_slopes_pval_plot, fd_types, characteristic[n], ['Annual', 'MAMJJA', 'SONDJF'], path = args.figure_path, savename = savename)

        # Repeat for different regions
        

    # Sensitivity analysis:
        # Determine standardized anomalies for multiple variables (SM, T, Tdew, PRES, WS, ET, PET, PRECIP, VPD when FD occurs, 1, 2, and 3 pentads ahead)
        # Correlate with FD occurence and other types of analyses (Nogeura et al. 2021, Mukherjee et al. 2022a)
        # Repeat for specific regions (may also connect type of climate anomalies to explain trends)
    if args.fd_sensitivity_analysis:
        variable_snames = ['tair', 'd2m', 'sp', 'ws', 'e', 'pev', 'tp', 'vpd', 'swvl1', 'swvl2', 'swvlrz']
        labels = ['T', r'T$_d$', 'Pres', 'WS', 'E', 'PE', 'Prec', 'VPD', 'SM1', 'SM2', 'RZSM']

        # Construct array of datetimes
        start = datetime(1979, 1, 1); end = datetime(2024, 12, 31)
        Ndays = (end - start).days
        dates_all = np.array([start + timedelta(days = day) for day in range(Ndays)])

        I, J = mask.shape
        mask_1d = mask.reshape(I*J)

        # Initialize the correlation dataset
        r = {}; sig = {}
        r_index = {}; sig_index = {}
        r_lag = {}; sig_lag = {}
        r_index_lag = {}; sig_index_lag = {}

        # Make the hypthesis testing for correlation
        rng = np.random.default_rng()
        test_method = stats.MonteCarloMethod(n_resamples = 5000, rvs = (rng.normal, rng.normal))

        # Make dictionaries for the energy/moisture limited analysis
        spi_anomalies = {}
        pet_anomalies = {}
        fd_starts = {}

        # Perform analysis for each FD method
        for fd_type, sname, fn_base, ind_sname, ind_base in zip(fd_types, fd_snames, fn_bases, index_snames, index_bases):
            if args.skip_variable_boxplots:
                # Collect datetimes of FD start; note the shape is lat x lon
                start_dates, start_dates_ind = load_start_times(args, fn_base, sname)

                variables = {}
                variables_5day = {}
                variables_10day = {}
                variables_15day = {}
                for var_sname in variable_snames:
                    # Load the variable
                    variable = load_raw_data(var_sname)

                    # Standardize the variable
                    variable = standardize_variable(variable, 
                                                    dates_all, 
                                                    datetime(1990, 1, 1),
                                                    datetime(2020, 12, 31))
                    
                    T, I, J = variable.shape
                    # NaN sea values
                    for t in range(T):
                        variable[t,:,:] = np.where(mask == 1, variable[t,:,:], np.nan)

                    variable = variable.reshape(T, I*J)
                    
                    # Collect the desired dates
                    variables[var_sname] = []
                    variables_5day[var_sname] = []
                    variables_10day[var_sname] = []
                    variables_15day[var_sname] = []
                    
                    # Note from this, all anomalies are in a flattened shape
                    for ij in tqdm(range(I*J), desc = 'Indexing %s anomalies'%var_sname):
                        # Skip sea and arid grids
                        if mask_1d[ij] == 0:
                            continue

                        # Collect the variable anomalies during start of FD
                        ind = start_dates_ind[ij]
                        ind = np.array(ind)
                        for i in ind:
                            variables[var_sname].append(variable[i,ij])

                        # Collect the variable anomalies 5 days before FD
                        ind = ind - 5
                        # ind = [i - 5 for i in ind]
                        if any(ind < 0): # Remove negative indices
                            zero_inds = np.where(ind < 0)[0]
                            ind = np.delete(ind, zero_inds) 
                        for i in ind:
                            variables_5day[var_sname].append(variable[i,ij])

                        # Collect variable anomalies 10 days before FD
                        ind = ind - 5
                        # ind = [i - 5 for i in ind]
                        if any(ind < 0): # Remove negative indices
                            zero_inds = np.where(ind < 0)[0]
                            ind = np.delete(ind, zero_inds) 
                        for i in ind:
                            variables_10day[var_sname].append(variable[i,ij])

                        # Collect variable anomalies 15 days before FD
                        ind = ind - 5
                        # ind = [i - 5 for i in ind]
                        if any(ind < 0): # Remove negative indices
                            zero_inds = np.where(ind < 0)[0]
                            ind = np.delete(ind, zero_inds) 
                        for i in ind:
                            variables_15day[var_sname].append(variable[i,ij])

                # Prepare the variable anomalies for plotting
                box_data = []
                box_data.append([variables_15day[var_sname] for var_sname in variable_snames])
                box_data.append([variables_10day[var_sname] for var_sname in variable_snames])
                box_data.append([variables_5day[var_sname] for var_sname in variable_snames])
                box_data.append([variables[var_sname] for var_sname in variable_snames])
                # print(box_data[0])

                # Make the box plot
                savename = 'variable_anomaly_boxplot_for_fd_%s_%s.png'%(fd_type, level)
                make_variable_boxplots(box_data, labels, 
                                    ['15 Days before FD', '10 Days before FD', '5 Days before FD', 'Start of FD'],
                                    fd_type, path = args.figure_path, savename = savename)
                
            if args.skip_correlation_plots:
                print(fd_type)
                # Load FD and index data for all years
                fd, fd_index = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask, return_fd_and_indices = True)

                # Reshape to time x space for easier calculations
                T, I, J = fd.shape
                fd = fd.reshape(T, I*J).astype(np.float32)
                fd_index = fd_index.reshape(T, I*J).astype(np.float32)

                # Make a spatial average lagged correlations
                fd_space = np.nanmean(fd, axis = -1)
                fd_index_space = np.nanmean(fd_index, axis = -1)

                # Perform correlation for each dataset
                for var_sname in variable_snames:
                    print(var_sname)
                    # Load the variable data
                    variable = load_raw_data(var_sname)

                    # Reshape for statistical calculations
                    variable = variable.reshape(T, I*J).astype(np.float32)

                    # Perform the correlation
                    # stat, _ = stats.pearsonr(fd, variable, axis = 0)
                    # pval = monte_carlo_significance(fd, variable, stat, N = 5000, statistic = 'correlation')
                    # stat = stat.reshape(I, J)
                    # pval = pval.reshape(I, J)
                    results = stats.pearsonr(fd, variable, method = test_method, axis = 0)
                    stat = results.statistic.reshape(I, J)
                    pval = results.pvalue.reshape(I, J)

                    # Fix longitude displacement
                    tmp = stat[:,lon_ind]
                    stat = np.concatenate([tmp, stat[:,:lon_ind[0]]], axis = 1)
                    tmp = pval[:,lon_ind]
                    pval = np.concatenate([tmp, pval[:,:lon_ind[0]]], axis = 1)

                    r['%s_%s'%(fd_type, var_sname)] = stat
                    sig['%s_%s'%(fd_type, var_sname)] = pval

                    print(r['%s_%s'%(fd_type, var_sname)].shape, sig['%s_%s'%(fd_type, var_sname)].shape)


                    # stat, _ = stats.pearsonr(fd_index, variable, axis = 0)
                    # pval = monte_carlo_significance(fd_index, variable, stat, N = 5000, statistic = 'correlation')
                    # stat = stat.reshape(I, J)
                    # pval = pval.reshape(I, J)
                    results = stats.pearsonr(fd_index, variable, method = test_method, axis = 0)

                    # Fix longitude displacement
                    stat = results.statistic.reshape(I, J)
                    pval = results.pvalue.reshape(I, J)

                    # Fix longitude displacement
                    tmp = stat[:,lon_ind]
                    stat = np.concatenate([tmp, stat[:,:lon_ind[0]]], axis = 1)
                    tmp = pval[:,lon_ind]
                    pval = np.concatenate([tmp, pval[:,:lon_ind[0]]], axis = 1)

                    r_index['%s_%s'%(fd_type, var_sname)] = stat
                    sig_index['%s_%s'%(fd_type, var_sname)] = pval
                    # r['%s_%s'%(fd_type, var_sname)] = correlate(fd, variable)
                    # r_index['%s_%s'%(fd_type, var_sname)] = correlate(fd_index, variable)

                    # Conduct significance testing
                    # sig['%s_%s'%(fd_type, var_sname)] = monte_carlo_significance(fd, variable, r['%s_%s'%(fd_type, var_sname)], N = 100, statistic = 'correlation')
                    # sig_index['%s_%s'%(fd_type, var_sname)] = monte_carlo_significance(fd_index, variable, r_index['%s_%s'%(fd_type, var_sname)], N = 100, statistic = 'correlation')

                    # Spatially average the variable for lag correlation
                    variable = np.nanmean(variable, axis = -1)

                    # Lag correlation
                    r_lag['%s_%s'%(fd_type, var_sname)] = []
                    r_index_lag['%s_%s'%(fd_type, var_sname)] = []
                    sig_lag['%s_%s'%(fd_type, var_sname)] = []
                    sig_index_lag['%s_%s'%(fd_type, var_sname)] = []
                    lags = np.arange(-30, 30, 1)
                    for n in tqdm(lags, desc = 'Determining Lag Correlations'):
                        N = np.abs(n)
                        if n < 0:
                            # lagged_corr, _ = stats.pearsonr(fd_space[N:], variable[:-N])
                            # lagged_index_corr, _ = stats.pearsonr(fd_index_space[N:], variable[:-N])
                            # lagged_sig = monte_carlo_significance(fd_space[N:], variable[:-N], lagged_corr, N = 5000, statistic = 'correlation')
                            # lagged_index_sig = monte_carlo_significance(fd_index_space[N:], variable[:-N], lagged_index_corr, N = 5000, statistic = 'correlation')
                            results = stats.pearsonr(fd_space[N:], variable[:-N], method = test_method)
                            results_index = stats.pearsonr(fd_index_space[N:], variable[:-N], method = test_method)
                            # lagged_corr = correlate(fd_space[N:].copy(), variable[:-N].copy())
                            # # Statistical significance test
                            # lagged_sig = monte_carlo_significance(fd_space[N:].copy(), variable[:-N].copy(), lagged_corr, N = 100, statistic = 'correlation')

                            # # Repeat with the index
                            # lagged_index_corr = correlate(fd_index_space[N:].copy(), variable[:-N].copy())
                            # # Statistical significance test
                            # lagged_index_sig = monte_carlo_significance(fd_index_space[N:].copy(), variable[:-N].copy(), lagged_index_corr, N = 100, statistic = 'correlation')
                        elif n == 0:
                            # lagged_corr, _ = stats.pearsonr(fd_space, variable)
                            # lagged_index_corr, _ = stats.pearsonr(fd_index_space, variable)
                            # lagged_sig = monte_carlo_significance(fd_space, variable, lagged_corr, N = 5000, statistic = 'correlation')
                            # lagged_index_sig = monte_carlo_significance(fd_index_space, variable, lagged_index_corr, N = 5000, statistic = 'correlation')
                            results = stats.pearsonr(fd_space, variable, method = test_method)
                            results_index = stats.pearsonr(fd_index_space, variable, method = test_method)
                            # lagged_corr = correlate(fd_space.copy(), variable.copy())
                            # # Statistical significance test
                            # lagged_sig = monte_carlo_significance(fd_space.copy(), variable.copy(), lagged_corr, N = 100, statistic = 'correlation')

                            # # Repeat with the index
                            # lagged_index_corr = correlate(fd_index_space.copy(), variable.copy())
                            # # Statistical significance test
                            # lagged_index_sig = monte_carlo_significance(fd_index_space.copy(), variable.copy(), lagged_index_corr, N = 100, statistic = 'correlation')
                        elif n > 0:
                            # lagged_corr = stats.pearsonr(fd_space[:-N], variable[N:])
                            # lagged_index_corr = stats.pearsonr(fd_index_space[:-N], variable[N:])
                            # lagged_sig = monte_carlo_significance(fd_space[:-N], variable[N:], lagged_corr, N = 5000, statistic = 'correlation')
                            # lagged_index_sig = monte_carlo_significance(fd_index_space[:-N], variable[N:], lagged_index_corr, N = 5000, statistic = 'correlation')
                            results = stats.pearsonr(fd_space[:-N], variable[N:], method = test_method)
                            results_index = stats.pearsonr(fd_index_space[:-N], variable[N:], method = test_method)
                            # lagged_corr = correlate(fd_space[:-N].copy(), variable[N:].copy())
                            # # Statistical significance test
                            # lagged_sig = monte_carlo_significance(fd_space[:-N].copy(), variable[N:].copy(), lagged_corr, N = 100, statistic = 'correlation')

                            # # Repeat with the index
                            # lagged_index_corr = correlate(fd_index_space[:-N].copy(), variable[N:].copy())
                            # # Statistical significance test
                            # lagged_index_sig = monte_carlo_significance(fd_index_space[:-N].copy(), variable[N:].copy(), lagged_index_corr, N = 100, statistic = 'correlation', acc = True)

                        lagged_corr = results.statistic; lagged_sig = results.pvalue
                        lagged_index_corr = results_index.statistic; lagged_index_sig = results_index.pvalue

                        # Add the lagged correlations to the list
                        r_lag['%s_%s'%(fd_type, var_sname)].append(lagged_corr)
                        r_index_lag['%s_%s'%(fd_type, var_sname)].append(lagged_index_corr)
                        sig_lag['%s_%s'%(fd_type, var_sname)].append(lagged_sig)
                        sig_index_lag['%s_%s'%(fd_type, var_sname)].append(lagged_index_sig)

                    r_lag['%s_%s'%(fd_type, var_sname)] = np.array(r_lag['%s_%s'%(fd_type, var_sname)])
                    r_index_lag['%s_%s'%(fd_type, var_sname)] = np.array(r_index_lag['%s_%s'%(fd_type, var_sname)])
                    sig_lag['%s_%s'%(fd_type, var_sname)] = np.array(sig_lag['%s_%s'%(fd_type, var_sname)])
                    sig_index_lag['%s_%s'%(fd_type, var_sname)] = np.array(sig_index_lag['%s_%s'%(fd_type, var_sname)])

            if args.skip_energy_moisture_drivers:
                # Collect datetimes of FD start; note the shape is lat x lon
                start_dates, start_dates_ind = load_start_times(args, fn_base, sname)
                fd_starts[fd_type] = start_dates_ind

        if args.skip_energy_moisture_drivers:
            # Load precipitation
            precip = load_raw_data('tp')
            T, I, J = precip.shape

            lat_ind = np.where((lat[:,0] >= 5) & (lat[:,0] <= 10))[0]
            lon_ind = np.where((lon[0,:] >= 5) & (lon[0,:] <= 40))[0]
            print(lon_ind, lat_ind)

            # tmp = precip[:,lat_ind,:]
            # precip = tmp[:,:,lon_ind]

            # Calculate the SPI
            spi = calculate_spi(precip, dates_all)

            # Load the PET
            pet = load_raw_data('pev')
            pet = -1*pet# Convert so positive PET represents energy fluxed into the atmosphere

            # tmp = pet[:,lat_ind,:]
            # pet = tmp[:,:,lon_ind]

            # Determine the standardized anomaly of PET
            pet = standardize_variable(pet, 
                                       dates_all, 
                                       datetime(1990, 1, 1),
                                       datetime(2020, 12, 31))
            
            # Reshape to have the same spatial dimension as the start dates
            # T, I, J = spi.shape
            spi = spi.reshape(T, I*J)
            pet = pet.reshape(T, I*J)
            print(np.nanmin(pet), np.nanmax(pet), np.nanmean(pet))

            # Determine the average PET and SPI for each recorded FD
            for fd_type in fd_types:
                spi_fd = []
                pet_fd = []
                for ij in range(I*J):
                    # # Skip sea and arid grid points
                    if mask_1d[ij] == 0:
                        continue

                    # Average over start of FD - 30 days to start of FD (covers onset period plus a little time before)
                    ind_end = fd_starts[fd_type][ij]
                    for t in ind_end:
                        # If FD occurs at the near of the time series, start at the beginning of the time series
                        if (t - 30) < 0:
                            start = 0
                        else:
                            start = t - 30

                        # Perform the average
                        spi_avg = np.nanmean(spi[start:t+1,ij])
                        pet_avg = np.nanmean(pet[start:t+1,ij])
                        
                        # Add data to the lists
                        spi_fd.append(spi_avg)
                        pet_fd.append(pet_avg)

                # Convert lists to array to allow finding conditions
                spi_fd = np.array(spi_fd)
                pet_fd = np.array(pet_fd)
                print(np.nanmin(pet_fd), np.nanmax(pet_fd), np.nanmean(pet_fd))
                print(pet_fd)

                spi_anomalies[fd_type] = spi_fd
                pet_anomalies[fd_type] = pet_fd

        
        if args.skip_correlation_plots:
            for n, var_sname in enumerate(variable_snames):
                map_data = [r['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                sig_data = [sig['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                
                # Make the maps
                savename = '%s_fd_%s_correlation_map.png'%(var_sname, level)
                make_correlation_maps(map_data, sig_data, lat, lon, fd_types, labels[n], path = args.figure_path, savename = savename)

                map_data = [r_index['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                sig_data = [sig_index['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                
                # Make the maps
                savename = '%s_fd_index_%s_correlation_map.png'%(var_sname, level)
                make_correlation_maps(map_data, sig_data, lat, lon, fd_types, labels[n], index_corr = True, path = args.figure_path, savename = savename)

            # Make the time series plots
            savename = 'lagged_correlation_fd_%s.png'%(level)
            make_lagged_correlation_plot(r_lag, sig_lag, lags, variable_snames, fd_types, labels, path = args.figure_path, savename = savename)

            # Make the time series plots
            savename = 'lagged_correlation_fd_index_%s.png'%(level)
            make_lagged_correlation_plot(r_index_lag, sig_index_lag, lags, variable_snames, fd_types, labels, path = args.figure_path, savename = savename)

        if args.skip_energy_moisture_drivers:

            # Make barplots with associated conditions
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
                moisture_limited_regime = np.nanstd(moisture_limited * 100/spi_anomalies[fd_type].size)
                energy_limited_regime = np.nanstd(energy_limited * 100/spi_anomalies[fd_type].size)
                both_limited_regime = np.nanstd(both_limited * 100/spi_anomalies[fd_type].size)
                moisture_condition_regime = np.nanstd(moisture_condition * 100/spi_anomalies[fd_type].size)
                energy_condition_regime = np.nanstd(energy_condition * 100/spi_anomalies[fd_type].size)

                bar_std.append([moisture_condition_regime, energy_condition_regime, moisture_limited_regime, energy_limited_regime, both_limited_regime])

            # Tick labels
            x_ticks = ['SPI<-1', 'PET>1', 'SPI<-1 &\nPET<1', 'SPI>-1 &\nPET>1', 'SPI<-1 &\nPET>1']

            savename = 'moisture_energy_driver_for_layer_%s_fd.png'%level
            make_barplots(bar_data, 
                          fd_types, 
                          'Relative Number (%) of FDs\nwith Given Conditions',
                          x_ticks,
                          bar_err = bar_std,
                          path = args.figure_path,
                          savename = savename)

        # EOFs?
        # Ways to determine energy vs. moisture driven FD (such as % FDs with precip anomaly < -1, and/or PET anomaly > 1; Fig. 4 in Christian et al. 2021)


    if args.fd_scatterplots:
        frequency = []
        frequency_sum = []
        frequency_win = []
        for fd_type, sname, fn_base, ind_sname, ind_base in zip(fd_types, fd_snames, fn_bases, index_snames, index_bases):
            # Perform FD statistics calculations for the all years
            freq, _, _ = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask)
            frequency.append(freq.flatten())

            freq, dur, sev = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask , times = 'summer')
            frequency_sum.append(freq.flatten())

            freq, dur, sev = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask , times = 'winter')
            frequency_win.append(freq.flatten())

        # Remove NaNs
        for i in range(len(frequency)):
            frequency[i] = np.delete(frequency[i], np.isnan(frequency[i]))
            frequency_sum[i] = np.delete(frequency_sum[i], np.isnan(frequency_sum[i]))
            frequency_win[i] = np.delete(frequency_win[i], np.isnan(frequency_win[i]))

        # In the event that removed NaNs cause length inconsistency
        ind = np.nanmin([len(frequency[0]), len(frequency[1]), len(frequency[2])]) # data[alt_ind is shorter]
        frequency[0] = frequency[0][:ind]; frequency[1] = frequency[1][:ind]; frequency[2] = frequency[2][:ind]

        ind = np.nanmin([len(frequency_sum[0]), len(frequency_sum[1]), len(frequency_sum[2])]) # data[alt_ind is shorter]
        frequency_sum[0] = frequency_sum[0][:ind]; frequency_sum[1] = frequency_sum[1][:ind]; frequency_sum[2] = frequency_sum[2][:ind]

        ind = np.nanmin([len(frequency_win[0]), len(frequency_win[1]), len(frequency_win[2])]) # data[alt_ind is shorter]
        frequency_win[0] = frequency_win[0][:ind]; frequency_win[1] = frequency_win[1][:ind]; frequency_win[2] = frequency_win[2][:ind]


        # Calculate correlation, pval, and regression parameters
        correlations = np.ones((len(frequency), 3))
        pvals = np.ones((len(frequency), 3))
        slopes = np.ones((len(frequency), 3))
        intercepts = np.ones((len(frequency), 3))

        # Make the hypthesis testing for correlation
        rng = np.random.default_rng()
        test_method = stats.MonteCarloMethod(n_resamples = 5000, rvs = (rng.normal, rng.normal))

        for i in range(len(frequency)):
            # Determine the other frequency to correlate the current one with
            alt_ind = i + 1
            if alt_ind >= len(frequency):
                alt_ind = 0

            # Perform correlation
            results = stats.pearsonr(frequency[i], frequency[alt_ind], method = test_method)
            correlations[i,0] = results.statistic
            pvals[i,0] = results.pvalue

            results = stats.pearsonr(frequency_sum[i], frequency_sum[alt_ind], method = test_method)
            correlations[i,1] = results.statistic
            pvals[i,1] = results.pvalue

            results = stats.pearsonr(frequency_win[i], frequency_win[alt_ind], method = test_method)
            correlations[i,2] = results.statistic
            pvals[i,2] = results.pvalue

            # Determine regression values
            slopes[i,0], intercepts[i,0], _ = least_squares(frequency[i], frequency[alt_ind])
            slopes[i,1], intercepts[i,1], _ = least_squares(frequency_sum[i], frequency_sum[alt_ind])
            slopes[i,2], intercepts[i,2], _ = least_squares(frequency_win[i], frequency_win[alt_ind])


        # Make the scatter plot 
        savename = 'fd_scatterplots_level_%s.png'%level
        make_scatterplots(frequency, 
                          frequency_sum, 
                          frequency_win, 
                          correlations, 
                          pvals, 
                          fd_types, 
                          ['Annual', 'MAMJJA', 'SONDJF'],
                          slope = slopes, 
                          intercept = intercepts,  
                          path = args.figure_path, 
                          savename = savename)
        # Scatter plots of different total FD events recoreded by different methods (Fig. 4 in Nogeura 2021)

    # Make map highlighting specific regions
    if args.make_region_map:
        savename = 'africa_regional_map.png'
        create_regional_boxes(savename = savename, path = args.figure_path)