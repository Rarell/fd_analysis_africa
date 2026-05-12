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
from make_figures import make_statistics_maps, make_boxplots, make_barplots, make_variable_boxplots, create_regional_boxes, make_trend_maps, timeseries_plot, make_correlation_maps, make_lagged_correlation_plot, make_scatterplots, make_errorbar_plot, make_eof_plot
from utils import subset_data, standardize_variable, calculate_spi

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
    fd_files = glob('%s/%s/%s*.nc'%(args.data_path, args.model, fn_base), recursive = True)
    fd_files = np.sort(fd_files)
    # print(fd_files)

    # Find all index files for severity
    index_files = glob('%s/%s/%s*.nc'%(args.data_path, args.model, index_base), recursive = True)
    index_files = np.sort(index_files)

    if ('rz' in fn_base) & (index_base is not None):
        sm_base1 = 'africa_volumetric_soil_water_layer_1_' if args.model == 'era5' else 'africa_gldas.soil_moisture_0-10cm.daily_'
        sm_base2 = 'africa_volumetric_soil_water_layer_2_' if args.model == 'era5' else 'africa_gldas.soil_moisture_10-40cm.daily_'
        index_files_1 = glob('%s/%s/liquid_vsm/%s*.nc'%(args.data_path, args.model, sm_base1), recursive = True)
        index_files_2 = glob('%s/%s/liquid_vsm/%s*.nc'%(args.data_path, args.model, sm_base2), recursive = True)
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
        fd_total.append(fd.astype(np.float32))
        time.append(dates)

        # Load one year of data of the appropiate index
        if ind_sname is not None:
            if (args.model == 'gldas') & ('swvl' in ind_sname):
                ind_sname = 'soilm'
            if 'rz' in index_base:
                index_files_combined = [index_files_1[t], index_files_2[t]]
                ind_snames = ['swvl1', 'swvl2'] if args.model == 'era5' else ['soilm', 'soilm']
                index_data = load_index_one_year(index_files_combined, ind_snames, index_base, times = times, I = I, J = J)
            else:
                index_data = load_index_one_year(index_files[t], ind_sname, index_base, times = times, I = I, J = J)
        else:
            # Placeholder so index_total can still be called when using FDII without 
            # significant code changes or further bloating the params argument
            index_data = np.zeros((1, I, J)) * np.nan 

        # Add to the total set of index data
        # print(np.nansum(fd > 0))
        index_total.append(index_data.astype(np.float32))

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
                                                                                          False if ind_sname is not None else True)

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
    if os.path.exists('%s/%s/%s'%(args.data_path, args.model, filename)):
        frequency, duration, severity = load_pickle('%s/%s/%s'%(args.data_path, args.model, filename))
        return frequency, duration, severity

    # Find all the files for the identified FD
    # Note files are data for 1 year of data
    fd_files = glob('%s/%s/%s*.nc'%(args.data_path, args.model, fn_base), recursive = True)
    fd_files = np.sort(fd_files)
    # print(fd_files)

    # Find all index files for severity
    index_files = glob('%s/%s/%s*.nc'%(args.data_path, args.model, index_base), recursive = True)
    index_files = np.sort(index_files)

    if ('rz' in fn_base) & (index_base is not None):
        sm_base1 = 'africa_volumetric_soil_water_layer_1_' if args.model == 'era5' else 'africa_gldas.soil_moisture_0-10cm.daily_'
        sm_base2 = 'africa_volumetric_soil_water_layer_2_' if args.model == 'era5' else 'africa_gldas.soil_moisture_10-40cm.daily_'
        index_files_1 = glob('%s/%s/liquid_vsm/%s*.nc'%(args.data_path, args.model, sm_base1), recursive = True)
        index_files_2 = glob('%s/%s/liquid_vsm/%s*.nc'%(args.data_path, args.model, sm_base2), recursive = True)
        index_files_1 = np.sort(index_files_1)
        index_files_2 = np.sort(index_files_2)

    # Load an FD file to initialize the frequency and duration datasets
    with Dataset(fd_files[0], 'r') as nc:
        fd = nc.variables[sname][:]

    # Load 40th percentile threshold (for severity calculations) if necessary
    if ind_sname is not None:
        with Dataset('%s/%s/africa_%s_40_percent_thresh.nc'%(args.data_path, args.model, ind_sname), 'r') as nc:
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
            if (args.model == 'gldas') & ('swvl' in ind_sname):
                ind_sname = 'soilm'
            if 'rz' in index_base:
                index_files_combined = [index_files_1[t], index_files_2[t]]
                ind_snames = ['swvl1', 'swvl2'] if args.model == 'era5' else ['soilm', 'soilm']
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
    save_pickle('%s/%s/%s'%(args.data_path, args.model, filename), [frequency, duration, severity], ['freq', 'dur', 'sev'])
        
    return frequency, duration, severity

def load_start_times(args, fn_base, sname, lat = None, lon = None, region = 'none'):
    '''
    Determine the start dates of flash drought for a given method
    '''

    # Find all the files for the identified FD
    fd_files = glob('%s/%s/%s*.nc'%(args.data_path, args.model, fn_base), recursive = True)
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

    # Subset if necessary
    if np.invert(region == 'none'):
        fd_total, _, _ = subset_data(fd_total, lat, lon, subset = region)
    
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
    pval = monte_carlo_significance(x.copy(), y.copy(), slope.copy(), N = 1000)

    # Reshape if necessary
    if len(y.shape) > 1:
        slope = slope.reshape(I, J)
        pval = pval.reshape(I, J)

    return slope, intercept, pval


def eof_analysis(args, fd_idx, mask, lat, lon, times, lon_ind, var_sname):
    T, I, J = fd_idx.shape
    fd_idx = fd_idx.reshape(T, I*J).astype(np.float32)
    mask2d = mask.reshape(I*J)

    # Detrend data
    print('Detrending data')
    t = np.arange(T)
    _, _, fd_trend = least_squares(t, fd_idx)
    fd_idx = fd_idx - fd_trend

    # Apply cosine latitude weighting and replace missing values with 0
    print('Applying weights')
    weights = np.sqrt(np.cos(np.pi * lat/180).reshape(I*J))
    fd_idx = fd_idx * weights[np.newaxis,:]

    # Replace missing values with 0
    print('Removing NaNs')
    fd_idx = np.delete(fd_idx, mask2d == 0, axis = -1)
    fd_idx = np.where(np.isnan(fd_idx), 0, fd_idx)

    # EOFs of each index variable
    print('Data prepared; performing SVD')
    # C = (np.dot(fd_idx.T, fd_idx)/T).astype(np.float32) # space x space matrix
    PCs, sig, EOFs = np.linalg.svd(fd_idx)
    eigval = (sig**2)/T
    # del C; gc.collect()

    # Standardize PCs
    print('Standardizing PCs and getting regression patterns')
    PCs = (PCs - np.nanmean(PCs, axis = 0))/np.nanstd(PCs, axis = 0)

    # Regress PCs onto EOFS
    regress_patterns = least_squares(PCs, fd_idx)
    print(regress_patterns.shape, eigval.size)

    # Make plots
    var_explained = eigval/np.nansum(eigval) * 100
    print(f'Variance explained for {var_sname} for first five modes: ', var_explained[:5])

    # Create the EOF plots (for the top three modes)
    for mode in range(1, 3+1):
        savename = '%s_eof_mode_%d.png'%(var_sname, mode)

        # Replace the NaN values and fill with regression pattern
        regress_map = np.zeros((I, J)) * np.nan
        regress_map = regress_map.reshape(I*J)
        k = 0
        for ij in range(I*J):
            if mask2d[ij] == 1:
                regress_map[ij] = regress_patterns[mode-1,k]
                k = k + 1
        regress_map = regress_map.reshape(I, J)

        # Correct longitude issues
        if args.model == 'era5':
            tmp = regress_map[:,lon_ind]
            regress_map = np.concatenate([tmp, regress_map[:,:lon_ind[0]]], axis = -1)

        # Perform a 90 day running mean on PC to smooth out the time series
        runmean = 90
        PCs[1:,mode-1] = np.convolve(PCs[1:,mode-1], np.ones((runmean))/runmean)[(runmean-1):]

        make_eof_plot(regress_map, 
                        lat, 
                        lon, 
                        PCs[1:,mode-1], 
                        times, 
                        var_sname, 
                        mode, 
                        var_explained[mode-1], 
                        path = '%s/%s'%(args.figure_path, args.model), 
                        savename = savename)

    # Significance plots
    N = eigval.size
    # Possible title: Eigen Value for Each Mode with Standard Error
    delta_lambda = eigval * np.sqrt(2/N)
    modes = np.arange(1, 15+1)
    savename = '%s_eigen_confidence_interval.png'%var_sname
    make_errorbar_plot(modes, eigval[:15], delta_lambda[:15], 'Modes', 'Eigen Values', path = '%s/%s'%(args.figure_path, args.model), savename = savename)

    # Remove larger files to free up space for next analysis
    del fd_idx, PCs, EOFs, sig, eigval, delta_lambda, regress_patterns
    gc.collect()
    

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
    parser.add_argument('--skip_eof_analysis', action = 'store_false', help = 'Skip the EOF analysis for driving variables')

    parser.add_argument('--em_lag_days', type = int, default = 0, help = 'Number of days before FD to examine SPI and PET for energy and moisture (EM) drivers')
    parser.add_argument('--model', type = str, default = 'era5', help = 'Type of reanalysis data examined (era5 or gldas)')
    parser.add_argument('--level', type = int, default = 0, help = 'Soil moisture level (must be 0 - 4; 0 means root zone depth)')
    parser.add_argument('--region', type = str, default = 'none', help = 'Specific subregion to focus on (valid: sahel, congo, easter, southern, madagascar)')
    parser.add_argument('--start_year', type = int, default = 1979, help = 'First year in FD dataset')
    parser.add_argument('--end_year', type = int, default = 2024, help = 'Last year in FD dataset')
    parser.add_argument('--nprocesses', type=int, default=1, help='Number of working threads for multiprocesses tasks')

    # Parse the arguments
    args = parser.parse_args()

    # Define the soil moisture level:
    level = 'rz' if args.level == 0 else str(args.level)

    if args.model == 'gldas':
        if args.level == 0:
            level_desc = 'rz'
        elif args.level == 1:
            level_desc = '0-10cm'
        elif args.level == 2:
            level_desc = '10-40cm'
        elif args.level == 3:
            level_desc = '40-100cm'
        elif args.level == 4:
            level_desc = '100-200cm'

    # Types of FD analysis; sesr = Christian et al. 2023 method 
    #                       rzsm = Yuan et al. 2019 method 
    #                       fdii = Otkin et al. 2021 method
    fd_types = ['sesr', 'rzsm', 'fdii']

    # Define the .nc keys for the FD datasets
    fd_snames = ['fd', 'fd%s'%level, 'fdii%s'%level]
    index_snames = ['sesr', 'swvl%s'%level, None] # if args.model == 'era5' else 'soilm'

    # Define the base of the filename for the FD datasets
    fn_bases = [
        'identified_fd/africa_fd_sesr_',
        'identified_fd/africa_fd_sm_%s_'%level,
        'fd_indices/africa_fdii_%s_'%level
    ]

    index_bases = [
        'fd_indices/africa_sesr_',
        'liquid_vsm/africa_volumetric_soil_water_layer_%s_'%level if args.model == 'era5' else 'liquid_vsm/africa_gldas.soil_moisture_%s.daily_'%level_desc,
        None
    ]

    # Load test dataset to obtain lats and lons
    with Dataset('%s/%s/%s2000.nc'%(args.data_path, args.model, fn_bases[0]), 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

    # Load the mask
    with Dataset('%s/%s/aridity_mask.nc'%(args.data_path, args.model), 'r') as nc:
        mask = nc.variables['aim'][0,:,:]

    if args.model == 'era5':
        lon_ind = np.where(lon[0,:] > 330)[0]
        lon_tmp = lon[:,lon_ind]
        lon = np.concatenate([lon_tmp, lon[:,:lon_ind[0]]], axis = 1)

        sub_lat = lat[::-1,0]
    else:
        sub_lat = lat[:,0]

    # Subset the mask if necessary
    if np.invert(args.region == 'none'):
            mask, _, _ = subset_data(mask, sub_lat, lon[0,:], subset = args.region)

    region = '' if args.region == 'none' else '_%s'%args.region

    climate_indices = ['enso', 'iod', 'mjo']

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

            if args.model == 'era5':

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

            if args.model == 'era5':

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

            if args.model == 'era5':

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
        make_statistics_maps(frequencies, lat, lon, 'frequency', fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = 0, cmax = 50, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        durations = [duration, duration_sum, duration_win]
        savename = 'duration_maps_%s.png'%level
        make_statistics_maps(durations, lat, lon, 'duration', fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = 0, cmax = 50, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        # For comparative analysis with different indices, normalize the severities
        max_sev = [np.nanmax(np.abs(severity[n])) for n in range(len(fd_types))]
        severity = [severity[n]/max_sev[n] for n in range(len(fd_types))]
        severity_sum = [severity_sum[n]/max_sev[n] for n in range(len(fd_types))]
        severity_win = [severity_win[n]/max_sev[n] for n in range(len(fd_types))]
        severities = [severity, severity_sum, severity_win]
        savename = 'severity_maps_%s.png'%level
        make_statistics_maps(severities, lat, lon, 'severity', fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = -1, cmax = 0, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)
        
        # Make box plots of statistics
        box_data = [[frequency[n].flatten(), frequency_sum[n].flatten(), frequency_win[n].flatten()] for n in range(len(fd_types))]
        savename = 'frequency_boxplots_%s.png'%level
        make_boxplots(box_data, fd_types, ['Annual', 'MAMJJA', 'SONDJF'], 'frequency', path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        box_data = [[duration[n].flatten(), duration_sum[n].flatten(), duration_win[n].flatten()] for n in range(len(fd_types))]
        savename = 'duration_boxplots_%s.png'%level
        make_boxplots(box_data, fd_types, ['Annual', 'MAMJJA', 'SONDJF'], 'duration', path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        box_data = [[severity[n].flatten(), severity_sum[n].flatten(), severity_win[n].flatten()] for n in range(len(fd_types))]
        savename = 'severity_boxplots_%s.png'%level
        make_boxplots(box_data, fd_types, ['Annual', 'MAMJJA', 'SONDJF'], 'severity', path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        # Special maps of characteristics, rather than making columns time
        characteristics = [frequency, duration, severity]
        savename = 'fd_characteristics_maps_%s.png'%level
        make_statistics_maps(characteristics, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        # "Summer" maps
        characteristics = [frequency_sum, duration_sum, severity_sum]
        savename = 'fd_characteristics_maps_%s_sum.png'%level
        make_statistics_maps(characteristics, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        # "Winter" maps 
        characteristics = [frequency_win, duration_win, severity_win]
        savename = 'fd_characteristics_maps_%s_win.png'%level
        make_statistics_maps(characteristics, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

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
            if args.model == 'era5':
                tmp = frequency[:,:,lon_ind]
                frequency = np.concatenate([tmp, frequency[:,:,:lon_ind[0]]], axis = 2)

                tmp = duration[:,:,lon_ind]
                duration = np.concatenate([tmp, duration[:,:,:lon_ind[0]]], axis = 2)

                tmp = severity[:,:,lon_ind]
                severity = np.concatenate([tmp, severity[:,:,:lon_ind[0]]], axis = 2)

            # Perform FD statistics calculations for the "summer"
            frequency_sum, duration_sum, severity_sum = calculate_fd_statistics_by_year(args, fn_base, sname, ind_base, ind_sname, mask, fd_type = fd_type, times = 'summer')
            if args.model == 'era5':
                tmp = frequency_sum[:,:,lon_ind]
                frequency_sum = np.concatenate([tmp, frequency_sum[:,:,:lon_ind[0]]], axis = 2)

                tmp = duration_sum[:,:,lon_ind]
                duration_sum = np.concatenate([tmp, duration_sum[:,:,:lon_ind[0]]], axis = 2)

                tmp = severity_sum[:,:,lon_ind]
                severity_sum = np.concatenate([tmp, severity_sum[:,:,:lon_ind[0]]], axis = 2)

            # Perform FD statistics calculations for the "winter"
            frequency_win, duration_win, severity_win = calculate_fd_statistics_by_year(args, fn_base, sname, ind_base, ind_sname, mask, fd_type = fd_type, times = 'winter')
            if args.model == 'era5':
                tmp = frequency_win[:,:,lon_ind]
                frequency_win = np.concatenate([tmp, frequency_win[:,:,:lon_ind[0]]], axis = 2)

                tmp = duration_win[:,:,lon_ind]
                duration_win = np.concatenate([tmp, duration_win[:,:,:lon_ind[0]]], axis = 2)

                tmp = severity_win[:,:,lon_ind]
                severity_win = np.concatenate([tmp, severity_win[:,:,:lon_ind[0]]], axis = 2)

            # Normalize the severity so the slope can be uniformly be displayed on maps and plots
            max_sev = np.nanmax(np.abs(severity))
            severity = severity/max_sev; severity_sum = severity_sum/max_sev; severity_win = severity_win/max_sev

            if args.region == 'none':
                if args.model == 'gldas':
                    # For unknown reasons, using GLDAS data past 2011 in the trend analysis produces 
                    # strange gridding artifacts that are not realistic; perform the trend only to 2011 for GLDAS
                    # and all for ERA5
                    ind = np.where(years <= 2011)[0]
                else:
                    ind = np.where(years <= 2025)[0]

                T, I, J = frequency.shape
                # Perform trend analysis
                slope_freq, _, pval_freq = trend_analysis(years[ind].copy(), frequency[ind,:].copy())
                slope_dur, _, pval_dur = trend_analysis(years[ind].copy(), duration[ind,:].copy())
                slope_sev, _, pval_sev = trend_analysis(years[ind].copy(), severity[ind,:].copy())

                T, I, J = frequency_sum.shape
                # Repeat for "summer"
                slope_freq_sum, _, pval_freq_sum = trend_analysis(years[ind].copy(), frequency_sum[ind,:].copy())
                slope_dur_sum, _, pval_dur_sum = trend_analysis(years[ind].copy(), duration_sum[ind,:].copy())
                slope_sev_sum, _, pval_sev_sum = trend_analysis(years[ind].copy(), severity_sum[ind,:].copy())

                T, I, J = frequency_win.shape
                # Repeat for "winter"
                slope_freq_win, _, pval_freq_win = trend_analysis(years[ind].copy(), frequency_win[ind,:].copy())
                slope_dur_win, _, pval_dur_win = trend_analysis(years[ind].copy(), duration_win[ind,:].copy())
                slope_sev_win, _, pval_sev_win = trend_analysis(years[ind].copy(), severity_win[ind,:].copy())

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
            if np.invert(args.region == 'none'):
                # Subset data to a region if necessary
                frequency, _, _ = subset_data(frequency, sub_lat, lon[0,:], subset = args.region)
                duration, _, _ = subset_data(duration, sub_lat, lon[0,:], subset = args.region)
                severity, _, _ = subset_data(severity, sub_lat, lon[0,:], subset = args.region)

                frequency_sum, _, _ = subset_data(frequency_sum, sub_lat, lon[0,:], subset = args.region)
                duration_sum, _, _ = subset_data(duration_sum, sub_lat, lon[0,:], subset = args.region)
                severity_sum, _, _ = subset_data(severity_sum, sub_lat, lon[0,:], subset = args.region)

                frequency_win, _, _ = subset_data(frequency_win, sub_lat, lon[0,:], subset = args.region)
                duration_win, _, _ = subset_data(duration_win, sub_lat, lon[0,:], subset = args.region)
                severity_win, _, _ = subset_data(severity_win, sub_lat, lon[0,:], subset = args.region)

                T, I, J = frequency.shape

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
        cmins = [-0.06, -0.5, -0.0015] # May need to adjust based on model
        cmaxes = [0.06, 0.5, 0.0015]   # May need to adjust based on model

        for n in range(len(characteristic)):
            # Make the maps
            if args.region == 'none':
                map_data = slopes[characteristic[n]]
                savename = '%s_trends_maps_%s.png'%(characteristic[n], level)
                make_trend_maps(map_data, lat, lon, characteristic[n], fd_types, ['Annual', 'MAMJJA', 'SONDJF'], cmin = cmins[n], cmax = cmaxes[n], path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

                # Plot the significance
                map_pval = pvals[characteristic[n]]
                savename = '%s_trends_significance_maps_%s.png'%(characteristic[n], level)
                make_trend_maps(map_data, lat, lon, characteristic[n], fd_types, ['Annual', 'MAMJJA', 'SONDJF'], sig = map_pval, cmin = 0, cmax = 3, significance_plots = True, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            # Make the time series plots
            savename = '%s_timeseries_trends_%s%s.png'%(characteristic[n], level, region)
            overall_ts_plot = [overall_ts[0+n], overall_ts[3+n], overall_ts[6+n]]
            overall_slopes_plot = [overall_slopes[0+n], overall_slopes[3+n], overall_slopes[6+n]]
            overall_intercepts_plot = [overall_intercepts[0+n], overall_intercepts[3+n], overall_intercepts[6+n]]
            overall_slopes_pval_plot = [overall_slopes_pval[0+n], overall_slopes_pval[3+n], overall_slopes_pval[6+n]]
            timeseries_plot(years, overall_ts_plot, overall_slopes_plot, overall_intercepts_plot, overall_slopes_pval_plot, fd_types, characteristic[n], ['Annual', 'MAMJJA', 'SONDJF'], path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

        # Special maps that have all characteristics instead of by time
        if args.region == 'none':
            map_data = [[slopes['frequency'][n][0], slopes['duration'][n][0], slopes['severity'][n][0]] for n in range(len(characteristic))]
            savename = 'characteristic_trends_maps_%s.png'%level
            make_trend_maps(map_data, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], cmin = cmins, cmax = cmaxes, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            map_pval = [[pvals['frequency'][n][0], pvals['duration'][n][0], pvals['severity'][n][0]] for n in range(len(characteristic))]
            savename = 'characteristic_trends_significance_maps_%s.png'%level
            make_trend_maps(map_data, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], sig = map_pval, cmin = 0, cmax = 3, significance_plots = True, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            # "Summer" maps
            map_data = [[slopes['frequency'][n][1], slopes['duration'][n][1], slopes['severity'][n][1]] for n in range(len(characteristic))]
            savename = 'characteristic_trends_maps_%s_sum.png'%level
            make_trend_maps(map_data, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], cmin = cmins, cmax = cmaxes, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            map_pval = [[pvals['frequency'][n][1], pvals['duration'][n][1], pvals['severity'][n][1]] for n in range(len(characteristic))]
            savename = 'characteristic_trends_significance_maps_%s_sum.png'%level
            make_trend_maps(map_data, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], sig = map_pval, cmin = 0, cmax = 3, significance_plots = True, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            # "Winter" map
            map_data = [[slopes['frequency'][n][2], slopes['duration'][n][2], slopes['severity'][n][2]] for n in range(len(characteristic))]
            savename = 'characteristic_trends_maps_%s_win.png'%level
            make_trend_maps(map_data, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], cmin = cmins, cmax = cmaxes, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            map_pval = [[pvals['frequency'][n][2], pvals['duration'][n][2], pvals['severity'][n][2]] for n in range(len(characteristic))]
            savename = 'characteristic_trends_significance_maps_%s_win.png'%level
            make_trend_maps(map_data, lat, lon, 'characteristics', fd_types, ['Frequency', 'Duration', 'Normalized Severity'], sig = map_pval, cmin = 0, cmax = 3, significance_plots = True, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)
        

    # Sensitivity analysis:
    if args.fd_sensitivity_analysis:
        if args.model == 'era5':
            variable_snames = ['tair', 'd2m', 'e', 'pev', 'tp', 'vpd', 'swvlrz', 'enso', 'iod', 'mjo']
            labels = ['T', r'T$_d$', 'E', 'PE', 'Prec', 'VPD', 'RZSM', 'ENSO', 'IOD/\nDMI', 'MJO']
        else:
            variable_snames = ['tair', 'd2m', 'e', 'pev', 'tp', 'vpd', 'swvlrz', 'enso', 'iod', 'mjo']
            labels = ['T', r'T$_d$', 'E', 'PE', 'Prec', 'VPD', 'RZSM', 'ENSO', 'IOD/\nDMI', 'MJO']
            # variable_snames = ['tair', 'sp', 'ws', 'e', 'pev', 'tp', 'swvlrz', 'enso', 'iod', 'mjo']
            # labels = ['T', 'Pres', 'WS', 'E', 'PE', 'Prec', 'RZSM', 'ENSO', 'IOD/\nDMI', 'MJO']

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
        test_method = stats.MonteCarloMethod(n_resamples = 1000, rvs = (rng.normal, rng.normal))

        # Make dictionaries for the energy/moisture limited analysis
        spi_anomalies = {}
        pet_anomalies = {}
        fd_starts = {}

        # Perform analysis for each FD method
        for fd_type, sname, fn_base, ind_sname, ind_base in zip(fd_types, fd_snames, fn_bases, index_snames, index_bases):
            if args.skip_variable_boxplots:
                # Collect datetimes of FD start; note the shape is lat x lon
                start_dates, start_dates_ind = load_start_times(args, fn_base, sname, lat = sub_lat, lon = lon[0,:], region = args.region)

                variables = {}
                variables_5day = {}
                variables_10day = {}
                variables_15day = {}
                for var_sname in variable_snames:
                    # Load the variable
                    variable = load_raw_data(var_sname, args.model)

                    if np.invert(args.region == 'none'):
                        variable, _, _ = subset_data(variable, sub_lat, lon[0,:], subset = args.region)

                    if (var_sname == 'pev') & (args.model == 'era5'):
                        variable = -1*variable # Convert so positive PET represents energy fluxed into the atmosphere

                    # Apply a 5 day running mean to bring pentad behavior
                    if var_sname not in climate_indices:
                        T = variable.shape[0]
                        runmean = 5

                        # Determine the appropriate start and end index for a centered running mean 
                        start_ind = int(np.round((runmean - 1)/2))
                        end_ind = int(T + runmean - 1 - start_ind)

                        # Apply running mean for each grid point
                        for i in range(I):
                            for j in range(J):
                                variable[:,i,j] = np.convolve(variable[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]

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
                savename = 'variable_anomaly_boxplot_for_fd_%s_%s%s.png'%(fd_type, level, region)
                make_variable_boxplots(box_data, labels, 
                                    ['15 Days before FD', '10 Days before FD', '5 Days before FD', 'Start of FD'],
                                    fd_type, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)
                
            if args.skip_correlation_plots:
                print(fd_type)
                
                # Load FD and index data for all years
                fd, fd_index = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask, return_fd_and_indices = True)

                # Subset if necessary
                if np.invert(args.region == 'none'):
                    fd, _, _ = subset_data(fd, sub_lat, lon[0,:], subset = args.region)
                    fd_index, _, _ = subset_data(fd_index, sub_lat, lon[0,:], subset = args.region)

                # Apply a 5 day running mean to smooth out white noise and deliver pentad behavior
                T = fd_index.shape[0]
                runmean = 5

                # Determine the appropriate start and end index for a centered running mean 
                start_ind = int(np.round((runmean - 1)/2))
                end_ind = int(T + runmean - 1 - start_ind)

                # Apply running mean for each grid point
                for i in range(I):
                    for j in range(J):
                        fd_index[:,i,j] = np.convolve(fd_index[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]

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
                    variable = load_raw_data(var_sname, args.model)

                    # Subset if necessary
                    if np.invert(args.region == 'none'):
                        variable, _, _ = subset_data(variable, sub_lat, lon[0,:], subset = args.region)

                    if (var_sname == 'pev') & (args.model == 'era5'):
                        variable = -1*variable # Convert so positive PET represents energy fluxed into the atmosphere

                    if var_sname not in climate_indices:
                        T = variable.shape[0]
                        # Apply a 5 day running mean to smooth out white noise and deliver pentad behavior
                        runmean = 5

                        # Determine the appropriate start and end index for a centered running mean 
                        start_ind = int(np.round((runmean - 1)/2))
                        end_ind = int(T + runmean - 1 - start_ind)

                        # Apply running mean for each grid point
                        for i in range(I):
                            for j in range(J):
                                variable[:,i,j] = np.convolve(variable[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]

                    # Reshape for statistical calculations
                    variable = variable.reshape(T, I*J).astype(np.float32)

                    # Perform the correlation
                    if args.region == 'none':
                        # Correlation analysis
                        results = stats.pearsonr(fd, variable, method = test_method, axis = 0)
                        stat = results.statistic.reshape(I, J)
                        pval = results.pvalue.reshape(I, J)

                        # Fix longitude displacement
                        if args.model == 'era5':
                            tmp = stat[:,lon_ind]
                            stat = np.concatenate([tmp, stat[:,:lon_ind[0]]], axis = 1)
                            tmp = pval[:,lon_ind]
                            pval = np.concatenate([tmp, pval[:,:lon_ind[0]]], axis = 1)

                        # Obtain the correlation and significance
                        r['%s_%s'%(fd_type, var_sname)] = stat
                        sig['%s_%s'%(fd_type, var_sname)] = pval

                        print(r['%s_%s'%(fd_type, var_sname)].shape, sig['%s_%s'%(fd_type, var_sname)].shape)

                        # Repeat for the index
                        results = stats.pearsonr(fd_index, variable, method = test_method, axis = 0)

                        # Fix longitude displacement
                        stat = results.statistic.reshape(I, J)
                        pval = results.pvalue.reshape(I, J)

                        # Fix longitude displacement
                        if args.model == 'era5':
                            tmp = stat[:,lon_ind]
                            stat = np.concatenate([tmp, stat[:,:lon_ind[0]]], axis = 1)
                            tmp = pval[:,lon_ind]
                            pval = np.concatenate([tmp, pval[:,:lon_ind[0]]], axis = 1)

                        r_index['%s_%s'%(fd_type, var_sname)] = stat
                        sig_index['%s_%s'%(fd_type, var_sname)] = pval

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
                            # Correlation analysis for negative lagged response
                            results = stats.pearsonr(fd_space[N:], variable[:-N], method = test_method)
                            results_index = stats.pearsonr(fd_index_space[N:], variable[:-N], method = test_method)

                        elif n == 0:
                            # Correlation for no lag
                            results = stats.pearsonr(fd_space, variable, method = test_method)
                            results_index = stats.pearsonr(fd_index_space, variable, method = test_method)

                        elif n > 0:
                            # Correlation for positive lagged response
                            results = stats.pearsonr(fd_space[:-N], variable[N:], method = test_method)
                            results_index = stats.pearsonr(fd_index_space[:-N], variable[N:], method = test_method)
                            
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
                start_dates, start_dates_ind = load_start_times(args, fn_base, sname, lat = sub_lat, lon = lon[0,:], region = args.region)
                fd_starts[fd_type] = start_dates_ind
                # print(fd_starts)

            if args.skip_eof_analysis:
                if args.model == 'gldas':
                    # Filler lon_ind fore the eof_analysis function
                    lon_ind = None
                    
                # Collect index data
                print('Loading %s and preparing data'%ind_sname)
                _, fd_idx = calculate_fd_statistics(args, fn_base, sname, ind_base, ind_sname, mask, return_fd_and_indices = True)
                ind_sname = 'fdii%s'%level if ind_sname is None else ind_sname

                eof_analysis(args, fd_idx, mask, lat, lon, dates_all, lon_ind, ind_sname)

                if ind_sname == 'fdii%s'%level:
                    # Also conduct analysis with PET, ET, P (SM was already done for Yuan/RZSM method)
                    analysis_vars = ['tp', 'pev', 'e']
                    for var in analysis_vars:
                        variable = load_raw_data(var, args.model)
                        eof_analysis(args, variable, mask, lat, lon, dates_all, lon_ind, ind_sname)

        if args.skip_energy_moisture_drivers:
            # Load precipitation
            precip = load_raw_data('tp', args.model)

            # Subset if necessary
            if np.invert(args.region == 'none'):
                precip, _, _ = subset_data(precip, sub_lat, lon[0,:], subset = args.region)

            if args.model == 'gldas':
                # Convert from kg m^-2 s^-2 to m day^-1 (division by density of water)
                precip = precip/1000 * 3600 * 24

            T, I, J = precip.shape

            # Calculate the SPI
            spi = calculate_spi(precip, dates_all)

            # Load the PET
            pet = load_raw_data('pev', args.model)

            # Subset if necessary
            if np.invert(args.region == 'none'):
                pet, _, _ = subset_data(pet, sub_lat, lon[0,:], subset = args.region)

            if args.model == 'era5':
                pet = -1*pet# Convert so positive PET represents energy fluxed into the atmosphere

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
                        if (t - args.em_lag_days) < 0:
                            start = 0
                        else:
                            start = t - args.em_lag_days

                        # Perform the average
                        spi_avg = np.nanmean(spi[start:t+1,ij])
                        pet_avg = np.nanmean(pet[start:t+1,ij])
                        
                        # Add data to the lists
                        # spi_fd.append(spi_avg)
                        # pet_fd.append(pet_avg)
                        spi_fd.append(spi[start,ij])
                        pet_fd.append(pet[start,ij])

                # Convert lists to array to allow finding conditions
                spi_fd = np.array(spi_fd)
                pet_fd = np.array(pet_fd)
                print(np.nanmin(spi_fd), np.nanmax(spi_fd), np.nanmean(spi_fd))
                print(spi_fd)

                spi_anomalies[fd_type] = spi_fd
                pet_anomalies[fd_type] = pet_fd


        # EOFs analysis
        if args.skip_eof_analysis:
            pass

            # This section of code is used to determine the RMSC
            # # Load the three indices
            # _, sesr = calculate_fd_statistics(args, fn_bases[0], fd_snames[0], index_bases[0], index_snames[0], mask, return_fd_and_indices = True)
            # _, sm = calculate_fd_statistics(args, fn_bases[1], fd_snames[1], index_bases[1], index_snames[1], mask, return_fd_and_indices = True)
            # _, fdii = calculate_fd_statistics(args, fn_bases[2], fd_snames[2], index_bases[2], index_snames[2], mask, return_fd_and_indices = True)

            # T, I, J = sesr.shape
            # sesr = sesr.reshape(T, I*J).astype(np.float32)
            # sm = sm.reshape(T, I*J).astype(np.float32)
            # fdii = fdii.reshape(T, I*J).astype(np.float32)

            # # Remove masked out values (reduce datasize)
            # mask2d = mask.reshape(I*J)
            # sesr = np.delete(sesr, mask2d == 0, axis = -1)
            # sm = np.delete(sm, mask2d == 0, axis = -1)
            # fdii = np.delete(fdii, mask2d == 0, axis = -1)

            # # Remove NaNs
            # sesr = np.where(np.isnan(sesr), 0, sesr)
            # sm = np.where(np.isnan(sm), 0, sm)
            # fdii = np.where(np.isnan(fdii), 0, fdii)
            
            # for var in variable_snames:
            #     # Load the variables
            #     variable = load_raw_data(var, args.model)
            #     variable = variable.reshape(T, I*J).astype(np.float32)
            #     variable = np.delete(variable, mask2d == 0, axis = -1)
            #     variable = np.where(np.isnan(variable), 0, variable)

            #     # Calculate the covariance between the variables and each index
            #     sesr_cov = np.cov(np.nanmean(sesr, axis = 1), np.nanmean(variable, axis = 1)) # np.cov delivers a covariance matrix; 
            #     sm_cov = np.cov(np.nanmean(sm, axis = 1), np.nanmean(variable, axis = 1))     # diagonals are the variance, off-diagonals are the covariance
            #     fdii_cov = np.cov(np.nanmean(fdii, axis = 1), np.nanmean(variable, axis = 1))

            #     # Calculate the RMSC
            #     sesr_rmsc = np.sqrt(sesr_cov[0,1]**2/(sesr_cov[0,0] * sesr_cov[1,1]))
            #     sm_rmsc = np.sqrt(sm_cov[0,1]**2/(sm_cov[0,0] * sm_cov[1,1]))
            #     fdii_rmsc = np.sqrt(fdii_cov[0,1]**2/(fdii_cov[0,0] * fdii_cov[1,1]))

            #     # Give RMSC values and determine if MCA is needed
            #     print('%s RMSC with SESR: %4.3f'%(var, sesr_rmsc))
            #     print('%s RMSC with SM%s: %4.3f'%(var, level, sm_rmsc))
            #     print('%s RMSC with FDII%s: %4.3f'%(var, level, fdii_rmsc))
            
        if args.skip_correlation_plots:
            if args.region == 'none':
                for n, var_sname in enumerate(variable_snames):
                    map_data = [r['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                    sig_data = [sig['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                    
                    # Make the maps
                    savename = '%s_fd_%s_correlation_map.png'%(var_sname, level)
                    make_correlation_maps(map_data, sig_data, lat, lon, fd_types, labels[n], path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

                    map_data = [r_index['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                    sig_data = [sig_index['%s_%s'%(fd_type, var_sname)] for fd_type in fd_types]
                    
                    # Make the maps
                    savename = '%s_fd_index_%s_correlation_map.png'%(var_sname, level)
                    make_correlation_maps(map_data, sig_data, lat, lon, fd_types, labels[n], index_corr = True, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            # Make the time series plots
            savename = 'lagged_correlation_fd_%s%s.png'%(level, region)
            make_lagged_correlation_plot(r_lag, sig_lag, lags, variable_snames, fd_types, labels, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

            # Make the time series plots
            savename = 'lagged_correlation_fd_index_%s%s.png'%(level, region)
            make_lagged_correlation_plot(r_index_lag, sig_index_lag, lags, variable_snames, fd_types, labels, path = '%s/%s/'%(args.figure_path, args.model), savename = savename)

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

            savename = 'moisture_energy_driver_%d_lag_day_for_layer_%s_fd%s.png'%(args.em_lag_days, level, region)
            make_barplots(bar_data, 
                          fd_types, 
                          'Relative Number (%) of FDs\nwith Given Conditions',
                          x_ticks,
                          bar_err = bar_std,
                          title = '%d Days before FD'%args.em_lag_days,
                          path = '%s/%s/'%(args.figure_path, args.model),
                          savename = savename)


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
            print(frequency[i].shape)
            print(frequency_sum[i].shape)
            print(frequency_win[i].shape)

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
                          path = '%s/%s/'%(args.figure_path, args.model), 
                          savename = savename)
        # Scatter plots of different total FD events recoreded by different methods (Fig. 4 in Nogeura 2021)

    # Make map highlighting specific regions
    if args.make_region_map:
        from pyhdf.SD import SD, SDC

        # Collect the land cover type data
        file = SD('../../modis/raw/MCD12C1.A2022001.061.2023244164746.hdf', SDC.READ)
        print(file.datasets())

        # Collect the data
        data_holder = file.select('Majority_Land_Cover_Type_1')
        print(data_holder.attributes())
        print(data_holder.info())

        # Land cover types
        # lct = data_holder.attributes().keys()
        lct = np.array([key for key in data_holder.attributes().keys()])
        print(lct[4:])

        data = data_holder.get()
        data = np.array(data)

        # Obtain the latitude and longitudes
        resolution = 0.05
        lat = np.arange(-90, 90, resolution)
        lat = lat[::-1]
        lon = np.arange(-180, 180, resolution)

        lat_ind = np.where((lat >= -42) & (lat <= 42))[0]
        lon_ind = np.where((lon >= -30) & (lon <= 55))[0]

        lat = lat[lat_ind]; lon = lon[lon_ind]
        data = data[lat_ind,:]
        data = data[:,lon_ind]
        lon, lat = np.meshgrid(lon, lat)

        # print(Dataset('../../modis/raw/MCD12C1.A2022001.061.2023244164746.hdf', 'r'))
        savename = 'africa_regional_map.png'
        create_regional_boxes(data, lat, lon, lct[4:], savename = savename, path = args.figure_path)

        # Close the file
        data_holder.endaccess()
        file.end()