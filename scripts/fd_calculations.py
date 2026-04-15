import os, sys, warnings
import gc
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
from scipy import stats
from netCDF4 import Dataset
from argparse import ArgumentParser
from scipy import interpolate
from scipy import signal
from tqdm import tqdm

# from inputs_outputs import load_nc

def calculate_climatology(
        e, 
        pet, 
        dates_all, 
        start,
        end,
        days_per_year: int = 366
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculates the climatological mean and standard deviation of ESR from daily ERA5 data.
    Climatological data is calculated for all grid points and for all timestamps in the year.

    Inputs:
    :param e: Evaporation dataset (np.ndarray, with shape time x lat x lon)
    :param pet: Potential evaporation dataset (np.ndarray with shape time x lat x lon)
    :param dates_all: Datetimes labels for each time step in e and pet (np.ndarray with shape time)
    :param start: Datetime of first date in the climatology to consider (e.g., Jan 1, 1980)
    :param end: Datetime of end date in climatology to consider (e.d., Dec 31, 2020)
    :param days_per_year: Total number of days in one year of data (use 366 if using daily data to include leap day)

    Outputs:
    :param means: Mean of ESR for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param stds: Standard deviations for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in means and stds (np.ndarray of shape time_for_one_year)
    '''
    

    T, I, J = e.shape
    T = days_per_year # Numbers of days in a year

    # Determine the indices for the climatology times
    clim_ind = np.where( (dates_all >= start) & (dates_all <= end) )[0]

    # All years in climatology calculations
    all_years = np.unique([date.year for date in dates_all[clim_ind]])
    years = np.array([date.year for date in dates_all[clim_ind]])

    days = np.array([day.day for day in dates_all[clim_ind]])
    months = np.array([day.month for day in dates_all[clim_ind]])

    # Get datetimes for one year (includes leap day)
    dates_year = np.array([datetime(2012,1,1) + timedelta(days = day) for day in range(days_per_year)])

    # Initialize climatology means and standard deviations + counts
    means = np.zeros((T, I, J), dtype = np.float32)
    stds = np.zeros((T, I, J), dtype = np.float32)

    N = np.zeros((T))

    # Construct ESR
    esr = e/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan 
    # print(np.nansum(np.isnan(esr)))

    print('Initialized variables, calculation means')

    # Conduct climatology calculations
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]

        # Sum over all all ESR in a given day
        tmp_sum = np.nansum(esr[ind,:,:], axis = 0)
        means[t,:,:] = np.nansum([means[t,:,:], tmp_sum], axis = 0) # np.nansum to account for any NaNs
        N[t] = N[t] + len(ind)

    # At the end, loop again to get to divide the sums by N to get the means
    for t, date in enumerate(dates_year):
        means[t,:,:] = means[t,:,:]/N[t]

    means = means.astype(np.float32)

    print('Means calculated, calculating standard deviations')
    
    # Loop over each day in the year for standard deviation (requires means)
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]
        
        # Sum over the squared errors in a given date
        error = np.nansum((esr[ind,:,:] - means[t,:,:])**2, axis = 0)
        stds[t,:,:] = np.nansum([stds[t,:,:], error], axis = 0)

    # One final loop to finish standard deviation calculations
    for t, date in enumerate(dates_year):
        stds[t,:,:] = np.sqrt(stds[t,:,:]/(N[t] - 1))

    stds = stds.astype(np.float32)
    print(np.min(means), np.max(means))
    print(np.min(stds), np.max(stds))

    print('Standard deviations calculated')

    # Create datetime labels for each timestep in the means and standard deviations
    ind = np.where(years == 2012)[0]
    one_year = dates_year # dates_all[ind]

    return means, stds, one_year

def calculate_percentile_thresholds(
        variable, 
        percentile, 
        dates_all, 
        start,
        end,
        days_per_year: int = 366
        ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculates the the percentile value of a variable from daily ERA5 data.
    Climatological data is calculated for all grid points and for all timestamps in the year.

    Inputs:
    :param variable: Variable dataset (np.ndarray, with shape time x lat x lon)
    :param percentile: Percentile threshold to determine
    :param dates_all: Datetimes labels for each time step in variable (np.ndarray with shape time)
    :param start: Datetime of first date in the climatology to consider (e.g., Jan 1, 1980)
    :param end: Datetime of end date in climatology to consider (e.d., Dec 31, 2020)
    :param days_per_year: Total number of days in one year of data (use 366 if using daily data to include leap day)

    Outputs:
    :param thresholds: Percentile thresholds for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in thresholds (np.ndarray of shape time_for_one_year)
    '''
    

    T, I, J = variable.shape
    T = days_per_year # Numbers of days in a year

    # Determine the indices for the climatology times
    clim_ind = np.where( (dates_all >= start) & (dates_all <= end) )[0]

    # All years in climatology calculations
    days = np.array([day.day for day in dates_all[clim_ind]])
    months = np.array([day.month for day in dates_all[clim_ind]])

    # Get datetimes for one year (includes leap day)
    dates_year = np.array([datetime(2012,1,1) + timedelta(days = day) for day in range(days_per_year)])

    # Initialize percentiles
    thresholds = np.zeros((T, I, J), dtype = np.float32)

    print('Initialized variables, calculating percentiles')

    # Conduct climatology calculations
    for t, date in tqdm(enumerate(dates_year)):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]
        
        thresholds[t,...] = np.nanpercentile(variable[ind,...], percentile, axis = 0)


    # Create datetime labels for each timestep in the means and standard deviations
    one_year = dates_year # dates_all[ind]

    return thresholds, one_year

def calculate_sesr(
        et, 
        pet, 
        dates, 
        means, 
        stds, 
        one_year
        ) -> np.ndarray:
    '''
    Calculate the standardized evaporative stress ratio (SESR) from ET and PET.
    
    Full details on SESR can be found in Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1.
    
    Inputs:
    :param et: Evapotranspiration (ET) dataset (np.ndarray of shape time x lat x lon)
    :param pet: Potential evapotranspiration (PET) dataset (np.ndarray of shape time x lat x lon)
    :param dates: Datetime labels for each timestep in et and pet (np.ndarray of shape time)
    :param means: Mean of ESR for each grid point and date in the year (np.ndarray of shape time_for_one_year x lat x lon)
    :param stds: Standard deviation of ESR for each grid point and date in the year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in means and stds (np.ndarray of shape time_for_one_year)
        
    Outputs:
    :param sesr: Calculate SESR (np.ndarray with shape time x lat x lon)
    '''

    # Get shape information
    T, I, J = et.shape
    # dates = np.array([datetime(year, 1, 1) + timedelta(days = t) for t in range(T)])


    # Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
    esr = et/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan
    # print(np.nansum(np.isnan(esr)))

    # Collect date information
    months = np.array([date.month for date in one_year])
    days = np.array([date.day for date in one_year])

    # Initialize SESR
    sesr = np.ones((T, I, J)) * np.nan

    for t, date in enumerate(dates):
        # Find the date index for the one year range
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        # Standardize the ESR to get SESR
        sesr[t,:,:] = (esr[t,:,:] - means[ind[0],:,:])/stds[ind[0],:,:]
            

    # Remove any unrealistic points
    sesr = np.where(sesr < -5, -5, sesr)
    sesr = np.where(sesr > 5, 5, sesr)

    sesr = sesr.astype(np.float32)
    
    print(np.nanmin(sesr), np.nanmax(sesr), np.nanmean(sesr))
    # print(np.sum(sesr <= -4.5), np.nansum(sesr >= 4.5))
    return sesr

def calculate_fdii(
        smp, 
        dates, 
        apply_runmean = True, 
        mask = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate the flash drought intensity index (FDII) from a soil moisture percentiles.
    FDII is on the same time scale as the input data.
    
    Full details on FDII can be found in Otkin et al. 2021: https://doi.org/10.3390/atmos12060741

    Note FDII can be calculated with the standardized soil moisture, or percentiles.
    Percentiles are used here for consistancy with Otkin et al. 2021
    
    Inputs:
    :param smp: Soil moisture percentile dataset (np.ndarray with shape time x lat x lon)
    :param year: Datetime labels for each timestep in smp (np.ndarray of shape time)
    :param apply_runmean: Apply a centered running mean (length 5) to the percentiles before FDII calculations (recommended for daily data)
    :param use_mask: Indicates whether to use a land-sea mask to improve computation speed
    :param mask: Land-sea mask with values 1 for land and 0 for sea (np.ndarray with shape lat x lon)

    Outputs:
    :param fdii: FDII drought index (np.ndarray with shape time x lat x lon)
    :param fd_int: The strength of the rapid intensification of the flash drought (np.ndarray with shape time x lat x lon)
    :param dro_sev: Severity of the drought component of the flash drought (np.ndarray with shape time x lat x lon)
    '''
       
    print('Initializing some variables')
    # Define some base constants
    PER_BASE = 15 # Minimum percentile drop for FD is 15 percentiles in 4 pentads
    T_BASE   = 4*5
    DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought
    
    T, I, J = smp.shape

    # Make the years, months, and/or days variables
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    # Apply a 5 day running mean requested by the user
    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5

        # Determine the appropriate start and end index for a centered running mean 
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)

        # Apply running mean for each grid point
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                smp[:,i,j] = np.convolve(smp[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]
    
    
    print(np.nanmin(smp), np.nanmax(smp))
    print(np.nanmean(smp))
    
    print('Calculating rapid intensification of flash drought')
    # Determine the rapid intensification based on percentile changes 
    # based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    fd_int = np.zeros((T, I, J))

    # Determine the intensification index
    # Note many time related values are multiplied by 5 to correspond to daily data instead of pentad
    for i in tqdm(range(I), desc = 'Calculating FD_INT'):
        for j in range(J):
            # Ignore sea points
            if mask[i,j] == 0: # ERA5 only
                continue
        
            for t in range(T-10): # Note the last two pentads are excluded as there is not enough time for significant SM drop
            
                obs = np.zeros((9*5)) # Note, the method detailed in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
                for nday in np.arange(2*5, 10*5+5, 1):
                    nday = int(nday)
                    if (t+nday) >= T: # If t + npend is in the future (beyond the dataset), break the loop and use 0s for obs instead
                        break         # This should only effect results in one November and December
                    else:
                        obs[nday-10] = (smp[t+nday,i,j] - smp[t,i,j])/nday # Note npend is the number of pentads the system is currently looking ahead to.

                # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), set FD_INT to 0.
                #  Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
                if np.nanmax(obs) < (PER_BASE/T_BASE):
                    fd_int[t,i,j] = 0
                else:
                    fd_int[t,i,j] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)
                
    print(np.min(fd_int), np.max(fd_int), np.mean(fd_int))
    
    
    print('Calculating drought severity')
    # Next determine the drought severity component using equation 2 in Otkin et al. 2021 
    # (and detailed in section 2.2 of the same paper)
    dro_sev = np.zeros((T, I, J)) # Initialize the first entry to 0, since there is no rapid intensification before it

    for i in tqdm(range(I), desc = 'Calculating DRO_SEV'):
        for j in range(J):
            
            # Ignore sea values
            if mask[i,j] == 0: # ERA5 only
                continue
            
            for t in range(1, T-5):
                if (fd_int[t,i,j] > 0):
                    
                    dro_sum = 0
                    for nday in np.arange(0, 18*5+5, 1): # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
                        
                        if (t+nday) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset 
                            dro_sev[t,i,j] = 0 # (this should only impact results near the end of the dataset)
                            break
                        else:
                            dro_sum = dro_sum + (DRO_BASE - smp[t+nday,i,j])
                            
                            if smp[t+nday,i,j] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if SM is no longer below the base percentile for drought
                                if nday < 4*5:
                                    # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after rapid intensificaiton (i.e., little to no impact)
                                    dro_sev[t,i,j] = 0
                                    break
                                else:
                                    dro_sev[t,i,j] = dro_sum/nday # Terminate the loop and determine the drought severity if the drought condition is broken
                                    break
                                
                            elif (nday >= 18*5): # Calculate the drought severity of the loop goes out 90 days, but the drought does not end
                                dro_sev[t,i,j] = dro_sum/nday
                                break
                            else:
                                pass
                
                # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately follow rapid intensification, drought is set 0
                else:
                    dro_sev[t,i,j] = 0
    
    print(np.min(dro_sev), np.max(dro_sev), np.mean(dro_sev))

    # Apply the mask
    for t in range(T):
        fd_int[t,:,:] = np.where(mask == 1, fd_int[t,:,:], np.nan)
        dro_sev[t,:,:] = np.where(mask == 1, dro_sev[t,:,:], np.nan)
    
    print('Calculating FDII')
    
    # Finally, FDII is the product of the rapid intensification and drought severity components
    fdii = fd_int * dro_sev
    
    print(np.min(fdii), np.max(fdii), np.mean(fdii))

    # Remove values less than 0
    fd_int[fd_int <= 0] = 0
    dro_sev[dro_sev <= 0] = 0
    fdii[fdii <= 0] = 0

    print('Done')
    
    return fdii, fd_int, dro_sev


def calculate_sm_percentiles(
        sm, 
        sm_all, 
        dates, 
        dates_all, 
        mask = None, 
        ) -> np.ndarray:
    '''
    Calculate the soil moisture percentiles using a larger popularion of soil moisture data

    Note this method is NOT space efficient. It requires in the full soil moisture dataset which 
    can be large to produce timely computations.

    Inputs:
    :param sm: Soil moisture dataset (np.ndarray with shape time x lat  x lon)
    :param sm_all: Full soil moisture dataset (np.ndarray with shape time_all_years x lat x lon)
    :param dates: Datetime labels for each time step in sm (np.ndarray with shape time)
    :param dates_all: Datetime labels for each time step in sm_all (np.ndarray with shape time_for_all_dates)
    :param mask: Land-sea mask with values 1 for land and 0 for sea (np.ndarray with shape lat x lon)
    
    Outputs:
    :param smp: Percentiles for each value in sm (np.ndarray with shape time x lat x lon)
    '''
        
    T, I, J= sm.shape # Obtain the dataset size to intialize the percentile dataset
    
    # All years in the time series
    all_years = np.unique([date.year for date in dates_all])

    # Calculate all years in the full time series
    years = np.array([date.year for date in dates_all])
    months = np.array([date.month for date in dates_all])
    days = np.array([date.day for date in dates_all])

        
    # Initialize percentile dataset
    smp = np.zeros((T, I, J))
 
    # sm = np.concatenate(sm, axis = 0)

    # n = 0
    for i in tqdm(range(I)):
        for j in range(J):
            # Skip sea values
            if mask[i,j] == 0:
                continue

            #print('%d/%d'%(n, I*J))
            sm_time_series = []

            # Determine the complete time series for a given grid point
            sm_time_series = sm_all[:,i,j]
            
            # Calculate the SM percentiles for all points in the time axis for a given grid point
            for t, date in enumerate(dates):
                # Obtain all indices for the current day of the year
                ind = np.where((date.day == days) & (date.month == months))[0] 

                # Calculate the SM percentile based on the current day of the year
                smp[t,i,j] = stats.percentileofscore(sm_time_series[ind], sm[t,i,j])

            # n = n+1

    print(np.nansum(smp <10), np.nansum(smp > 90))
    print(np.nanmin(smp), np.nanmax(smp), np.nanmean(smp))
    smp = smp.astype(np.float32)

    return smp


def christian_fd(
        sesr, 
        mask, 
        dates, 
        start_year = 1990, 
        end_year = 2020, 
        apply_runmean = False,
        years = None, 
        months = None, 
        days = None
        ) -> np.ndarray:
    '''
    Calculate the flash drought using an updated version of the method described in Christian et al. 2019
    (https://doi.org/10.1175/JHM-D-18-0198.1). This method uses the evaporative stress ratio (SESR) to 
    identify flash drought. Updates to the method are details (for LSWI) in Christian et al. 2022
    (https://doi.org/10.1016%2Fj.rsase.2022.100770).
    
    Inputs:
    :param sesr: Input SESR values, (np.ndarray with shape time x lat x lon)
    :param mask: Land-sea mask for the et and pet variables. A value of None can be provided to not use a mask
    :param dates: Array of datetimes corresponding to the timestamps in sesr (np.ndarray with shape time)
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param apply_runmean: Apply a centered running mean (length 5) to SESR before FD calculations (recommended for daily data)
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in sesr
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
        
    T, I, J = sesr.shape

    # Is a mask provided?
    mask_provided = mask is not None
        
    # Apply a 5 day running mean requested by the user
    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5

        # Determine the appropriate start and end index for a centered running mean 
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)

        # Apply running mean for each grid point
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                sesr[:,i,j] = np.convolve(sesr[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]

    # Initialize some variables
    sesr_inter = np.ones((T, I, J)) * np.nan
    sesr_filt  = np.ones((T, I, J)) * np.nan
    
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    
    # sesr = sesr.reshape(T, I*J, order = 'F')
    # sesr_inter = sesr_inter.reshape(T, I*J, order = 'F')
    # sesr_filt  = sesr_filt.reshape(T, I*J, order = 'F')
    
    # if mask_provided:
    #     mask = mask.reshape(I*J, order = 'F')
    
    x = np.arange(-6.5, 6.5, (13/T))#[:-1] # a variable covering the range of all SESR values with 1 entry for each time step
    print(x.size, T)
    
    # Parameters for the filter
    WinLength = 21*5 # Window length of 21 pentads
    PolyOrder = 4

    # Perform a basic linear interpolation for NaN values and apply a SG filter
    print('Applying interpolation and Savitzky-Golay filter to SESR')
    for i in tqdm(range(I), desc = 'Applying SG Filter'):
        for j in range(J):
            if mask_provided:
                if mask[i,j] == 0:
                    continue
                else:
                    pass
            
            # Perform a linear interpolation to remove NaNs
            ind = np.isfinite(sesr[:,i,j])
            if np.nansum(ind) == 0:
                continue
            else:
                pass
            
            ind = np.where(ind == True)[0]
            interp_func = interpolate.interp1d(x[ind], sesr[ind,i,j], kind = 'linear', fill_value = 'extrapolate')
            
            sesr_inter[:,i,j] = interp_func(x)
            
            # Apply the Savitzky-Golay filter to the interpolated SESR data
            sesr_filt[:,i,j] = signal.savgol_filter(sesr_inter[:,i,j], WinLength, PolyOrder)
        
    # Reorder SESR back to 3D data
    #sesr_filt = sesr_filt.reshape(T, I, J, order = 'F')



    # Determine the change in SESR
    print('Calculating the change in SESR')
    delta_sesr  = np.ones((T, I, J)) * np.nan
    
    delta_sesr[1:,:,:] = sesr_filt[1:,:,:] - sesr_filt[:-1,:,:]

    
    # Begin the flash drought calculations
    print('Identifying flash drought')
    fd = np.ones((T, I, J)) * np.nan

    #fd = fd.reshape(T, I*J, order = 'F')
    #sesr_filt = sesr_filt.reshape(T, I*J, order = 'F')
    #delta_sesr = delta_sesr.reshape(T, I*J, order = 'F')

    dsesr_percentile = 25
    sesr_percentile  = 20
    
    min_change = timedelta(days = 30)
    start_date = dates[-1]
    
    for i in tqdm(range(I), desc = 'Identifying FD'):
        for j in range(J):
            if mask_provided:
                if mask[i,j] == 0:
                    continue
            
            start_date = dates[-1]
            for t in range(T):
                ind = np.where( (dates[t].month == months[climo_index]) & (dates[t].day == days[climo_index]) )[0]
                
                # Determine the percentiles of dSESR and SESR
                ri_crit = np.nanpercentile(delta_sesr[ind,i,j], dsesr_percentile)
                dc_crit = np.nanpercentile(sesr_filt[ind,i,j], sesr_percentile)

                
                # If start_date != dates[-1], the rapid intensification criteria is satisified
                # If the rapid intensification and drought component criteria are satisified (and FD period is 30+ days)
                # then FD occurs
                if ( (dates[t] - start_date) >= min_change) & (sesr_filt[t,i,j] <= dc_crit):
                    fd[t,i,j] = 1
                else:
                    fd[t,i,j] = 0
                
                # # If the change in SESR is below the criteria, change the start date of the flash drought
                if (delta_sesr[t,i,j] <= ri_crit) & (start_date == dates[-1]):
                    start_date = dates[t]
                elif (delta_sesr[t,i,j] <= ri_crit) & (start_date != dates[-1]):
                    pass
                else:
                    start_date = dates[-1]

    # Apply the mask
    for t in range(T):
        fd[t,:,:] = np.where(mask == 1, fd[t,:,:], np.nan)
            
    # Re-order the flash drought back into a 3D array
    # fd = fd.reshape(T, I, J, order = 'F')
    print('Done')
    
    return fd


def yuan_fd(
        smp, 
        mask, 
        dates, 
        apply_runmean = False, 
        years = None, 
        months = None, 
        days = None
        ) -> np.ndarray:
    '''
    Calculate the flash drought using the method described in Yuan et al. 2019 (https://doi.org/10.1038/s41467-019-12692-7). 
    This method uses the soil moisture percentiles (SMP; typically 0 - 40 cm average) to identify flash drought.
    
    Inputs:
    :param smp: Input SM percentiles (np.ndarray with shape time x lat x lon)
    :param mask: Land-sea mask for the SM percentiles
    :param dates: Array of datetimes corresponding to the timestamps in smp
    :param apply_runmean: Apply a centered running mean (length 5) to SESR before FD calculations (recommended for daily data)
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in smp
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
        
    T, I, J = smp.shape

    # Apply a 5 day running mean requested by the user
    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5

        # Determine the appropriate start and end index for a centered running mean 
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)

        # Apply running mean for each grid point
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                smp[:,i,j] = np.convolve(smp[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]
    
        
    # Begin drought identification process
    print('Identifying flash droughts')
    fd = np.zeros((T, I, J)) * np.nan
    print(T)
  
    for i in tqdm(range(I), desc = 'Determining FD'):
        for j in range(J):
        
            if mask[i,j] == 0:
                continue
            
            for t in range(T-12*5): # Exclude the last few months in the dataset for simplicity since FD identification involves looking up to 12 pentads ahead
                # If FD analysis was already conducted (process involves looking ahead), skip the analysis
                if (fd[t,i,j] == 1) | (fd[t,i,j] == 0):
                    continue

                # First determine if the soil moisture is below the 40 percentile (FD possibly begins)
                if smp[t,i,j] <= 40:
                    rate = []

                    # Start looping up to 12 pentads (60 days) ahead
                    for p in range(1, 12*5):
                        # Determine the rate of percentile change
                        if (t + p >= T - 1):
                            rate.append(smp[t+p,i,j] - smp[-1,i,j])
                        else:
                            rate.append(smp[t+p-1,i,j] - smp[t+p,i,j])

                        # When the percentiles fall below the 20th percentile, drought begins
                        # Also another requirement for FD is the average range rate of SMP change must be >= 5 percentiles/pentad = 1 percentile/day
                        # (i.e., an average decrease of 1 percentile per day)
                        # print(rate)
                        if (smp[t+p,i,j] <= 20) & (np.nanmean(rate) >= 1):
                            # Continue looking forward to when the SM percentiles are above 20 (drought ends/recover begins)
                            # Note the indices collected should equate to the number of days, after SMP < 20 to when SMP > 20 (only true for daily data)
                            drought_recover = np.where(smp[t+p:,i,j] > 20)[0]
                            # print(smp[t+p:t+12*5,i,j])
                            # print(drought_recover)
                            # print(smp[t+p:,i,j].shape, smp[t+p:,i,j])

                            # Unique case near the end of the time series; since no points were found to end drought at the end,
                            # label all points from t+p onwards as FD
                            if (len(drought_recover) < 1) | ((t+p) > (T - 12*5)):
                                fd[t:t+p,i,j] = 0

                                fd[t+p:,i,j] = 1
                                break

                            # The last requirement for FD: The drought must last for 15+ days
                            # FD ends on the first instance when SM percentiles > 20 (so first entry in drought recovery)
                            if drought_recover[0] >= 15:
                                # Intensification period is labeled as non-FD (FD is still developing)
                                fd[t:t+p,i,j] = 0

                                # Label all days when SM percentiles < 20 as FD
                                fd[t+p:t+p+drought_recover[0],i,j] = 1

                                # Analysis is concluded; break the loop that is looking ahead 
                                # (all time points in this event should be labeled FD)
                                break

                            elif drought_recover[0] < 15:
                                # Event was too short to be impactful and thus classified as FD
                                fd[t:t+p+drought_recover[0],i,j] = 0
                        
                        # If drought condition is reached, but the decline was not rapid enough for FD, 
                        # the event drought is labeled as non-FD
                        elif (smp[t+p,i,j] <= 20) & (np.nanmean(rate) < 1):
                            drought_recover = np.where(smp[t+p:,i,j] > 20)[0]

                            # Unique case near the end of the time series; simply break the loop 
                            # (leftover NaNs will be turned to 0 later)
                            if (len(drought_recover) < 1) | ((t+p) > (T - 12*5)):
                                break

                            fd[t:t+p+drought_recover[0],i,j] = 0

                            # Conclude analysis for current event
                            break

                # SM percentiles above the 40th percentile (no occurrence of FD)
                else:
                    fd[t,i,j] = 0
        
        # print(np.nansum(fd[:,i,:]))
    
    # Sea values, and remaining days in the last 60 days of the dataset are labeled as non-FD
    fd[np.isnan(fd)] = 0

    # Apply the mask
    for t in range(T):
        fd[t,:,:] = np.where(mask == 1, fd[t,:,:], np.nan)

    print('Done')
    
    return fd


if __name__ == '__main__':
    description = 'Create indices and perform FD calculates over the African domain'
    parser = ArgumentParser(description = description)
    parser.add_argument('--load_et_and_pet_data', action = 'store_true', help = 'Load in the full set ET and PET data for all layers')
    parser.add_argument('--load_sm_data', action = 'store_true', help = 'Load in the full set of SM data for all layers')
    parser.add_argument('--load_smp_data', action = 'store_true', help = 'Load in the full set of SM percentile data for all layers')
    parser.add_argument('--calculate_thresholds', action = 'store_true', help = 'Calculate 40th percentile thresholds')
    parser.add_argument('--calculate_sesr', action = 'store_true', help = 'Calculate the ESR climatologies, and calculate SESR and save as nc files')
    parser.add_argument('--calculate_fd_sesr', action = 'store_true', help = 'Identify FD based on SESR as outlined in Christian et al. 2023')
    parser.add_argument('--calculate_sm_percentiles', action = 'store_true', help = 'Calculate soil moisture percentiles based for all SM layers')
    parser.add_argument('--calculate_fdii', action = 'store_true', help = 'Calculate FDII and save as nc files')
    parser.add_argument('--calculate_fd_yuan', action = 'store_true', help = 'Identify FD based on SM according to the Yuan et al. 2019')

    # parser.add_argument('--year', type = int, default = 0, help = 'Year to perform percentile/index calculations (note the actual year used is year + 2000)')
    parser.add_argument('--level', type = int, default = 0, help = 'ERA5 soil moisture level (must be 0 - 4; 0 means root zone depth)')
    parser.add_argument('--model', type = str, default = 'era5', help = 'Type of reanalysis model (era5 or gldas) to perform analysis for')

    args = parser.parse_args()

    days_per_year = 365

    # Define the base path to the datasets
    base_path = '/ourdisk/hpc/ai2es/sedris/fd_analysis/data'

    # Determine all years and build datetimes
    all_years = np.arange(1979, 2024+1)

    start = datetime(1979, 1, 1)
    N_leap_days = np.sum((all_years % 4) == 0) # Note datetimes follow the scheme of a leap day every 4 years, even if it isn't exactly correct
    T_full = (days_per_year * all_years.size) + N_leap_days
    dates = np.array([start + timedelta(days = t) for t in range(T_full)])

    # Collect an array of all year values
    years = np.array([date.year for date in dates])

    # Load the aridity mask
    with Dataset('%s/%s/aridity_mask.nc'%(base_path, args.model), 'r') as nc:
        mask = nc.variables['aim'][0,:,:]

    # Load all ET, PET
    if args.load_et_and_pet_data:
        print('Loading ET and PET')
        if args.model == 'era5':
            base_et_fn = 'africa_evaporation'
            base_pet_fn = 'africa_potential_evaporation'
            sname_et = 'e'; sname_pet = 'pev'
        else:
            base_et_fn = 'africa_gldas.evaporation.daily'
            base_pet_fn = 'africa_gldas.potential_evaporation.daily'
            sname_et = 'evap'; sname_pet = 'pevap'

        # Initialize lists
        et = []; pet = []
        for year in all_years:
            # Load ET data
            with Dataset('%s/%s/evaporation/%s_%04d.nc'%(base_path, args.model, base_et_fn, year), 'r') as nc:
                tmp = nc.variables[sname_et][:]

                # Also load lat and lon information here
                lat = nc.variables['lat'][:]
                lon = nc.variables['lon'][:]

                # Add ET data to the list
                et.append(tmp)

            # Load PET data
            with Dataset('%s/%s/potential_evaporation/%s_%04d.nc'%(base_path, args.model, base_pet_fn, year), 'r') as nc:
                tmp = nc.variables[sname_pet][:]

                # Add PET data to the list
                pet.append(tmp)                

        # Convert to arrays
        et = np.concatenate(et)
        pet = np.concatenate(pet)

        # For GLDAS, some unit conversion is needed for consistency with ET
        if args.model == 'gldas': 
            pet = pet / (2.5e6) # Division by latent heat of vaporization yields conversion of W m^-2 = J s^-1 m^-2 -> kg s^-1 m^-2

    
    if args.load_sm_data:
        if args.model == 'era5':
            base_sm1_fn = 'africa_volumetric_soil_water_layer_1'
            base_sm2_fn = 'africa_volumetric_soil_water_layer_2'
            base_sm3_fn = 'africa_volumetric_soil_water_layer_3'
            base_sm4_fn = 'africa_volumetric_soil_water_layer_4'
            sname1 = 'swvl1'; sname2 = 'swvl2'; sname3 = 'swvl3'; sname4 = 'swvl4'
        else:
            base_sm1_fn = 'africa_gldas.soil_moisture_0-10cm.daily'
            base_sm2_fn = 'africa_gldas.soil_moisture_10-40cm.daily'
            base_sm3_fn = 'africa_gldas.soil_moisture_40-100cm.daily'
            base_sm4_fn = 'africa_gldas.soil_moisture_100-200cm.daily'
            sname1 = sname2 = sname3 = sname4 = 'soilm'

        # Initialize lists
        sm = {}
        sm[1] = []; sm[2] = []; sm[3] = []; sm[4] = []
        for year in all_years:              
            # Load SM data
            with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, args.model, base_sm1_fn, year), 'r') as nc:
                tmp = nc.variables[sname1][:]

                # Also load lat and lon information here
                lat = nc.variables['lat'][:]
                lon = nc.variables['lon'][:]

                sm[1].append(tmp)

            with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, args.model, base_sm2_fn, year), 'r') as nc:
                tmp = nc.variables[sname2][:]

                sm[2].append(tmp)

            with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, args.model, base_sm3_fn, year), 'r') as nc:
                tmp = nc.variables[sname3][:]

                sm[3].append(tmp)

            with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, args.model, base_sm4_fn, year), 'r') as nc:
                tmp = nc.variables[sname4][:]

                sm[4].append(tmp)

        # Convert to arrays
        sm[1] = np.concatenate(sm[1]); sm[2] = np.concatenate(sm[2]); sm[3] = np.concatenate(sm[3]); sm[4] = np.concatenate(sm[4])

    # Load SM percentiles
    if args.load_smp_data:
        # Define the key
        key = 'rz' if args.level == 0 else args.level

        # Initialize lists
        smp = {}
        smp[key] = []
        # smp[1] = []; smp[2] = []; smp[3] = []; smp[4] = []; smp['rz'] = []
        for year in all_years:              
            # Load SM data
            with Dataset('%s/%s/soil_moisture_percentiles/africa_soil_moisture_percentiles_%s_%04d.nc'%(base_path, args.model, str(key), year), 'r') as nc:
                tmp = nc.variables['smp%s'%str(key)][:]

                # Also load lat and lon information here
                lat = nc.variables['lat'][:]
                lon = nc.variables['lon'][:]

                smp[key].append(tmp)
            # with Dataset('%s/soil_moisture_percentiles/africa_soil_moisture_percentiles_1_%04d.nc'%(base_path, year), 'r') as nc:
            #     tmp = nc.variables['smp1'][:]

            #     # Also load lat and lon information here
            #     lat = nc.variables['lat'][:]
            #     lon = nc.variables['lon'][:]

            #     smp[1].append(tmp)

            # with Dataset('%s/soil_moisture_percentiles/africa_soil_moisture_percentiles_2_%04d.nc'%(base_path, year), 'r') as nc:
            #     tmp = nc.variables['smp2'][:]

            #     smp[2].append(tmp)

            # with Dataset('%s/soil_moisture_percentiles/africa_soil_moisture_percentiles_3_%04d.nc'%(base_path, year), 'r') as nc:
            #     tmp = nc.variables['smp3'][:]

            #     smp[3].append(tmp)

            # with Dataset('%s/soil_moisture_percentiles/africa_soil_moisture_percentiles_4_%04d.nc'%(base_path, year), 'r') as nc:
            #     tmp = nc.variables['smp4'][:]

            #     smp[4].append(tmp)

            # with Dataset('%s/soil_moisture_percentiles/africa_soil_moisture_percentiles_rz_%04d.nc'%(base_path, year), 'r') as nc:
            #     tmp = nc.variables['smprz'][:]

            #     smp['rz'].append(tmp)

        # Convert to arrays
        # smp[1] = np.concatenate(smp[1]); smp[2] = np.concatenate(smp[2]); smp[3] = np.concatenate(smp[3]); smp[4] = np.concatenate(smp[4]); smp['rz'] = np.concatenate(smp['rz'])
        smp[key] = np.concatenate(smp[key])

    # Calculate SESR and save the results
    if args.calculate_sesr:
        # Calculate ET and PET climatologies for the 1990 - 2020 30 year period
        print('Calculating climatology')
        esr_means, esr_stds, one_year = calculate_climatology(et, 
                                                              pet, 
                                                              dates, 
                                                              datetime(1990, 1, 1),
                                                              datetime(2020, 12, 31),
                                                              days_per_year = 366)
        print(one_year)
        # Calculate SESR for each year and save the results
        print('Calculating SESR')
        for year in all_years:
            print(year)
            # Perform calculations for 1 year at a time (data saved as yearly files)
            ind = np.where(year == years)[0]

            et_year = et[ind,:,:]; pet_year = pet[ind,:,:]; dates_year = dates[ind]

            # Calculate SESR
            sesr = calculate_sesr(et_year, 
                                  pet_year, 
                                  dates_year, 
                                  esr_means, 
                                  esr_stds, 
                                  one_year)
            
            # Convert timestamps to a string for saving in .nc file
            time = np.array([date.isoformat() for date in dates_year])

            # Save the results
            with Dataset('%s/%s/africa_sesr_%04d.nc'%(base_path, args.model, year), 'w', format = 'NETCDF4') as nc:
            # with Dataset('../era5/sesr_%04d.nc'%(year), 'w', format = 'NETCDF4') as nc:
                nc.description = 'Daily %s reanalysis data for SESR over Africa, calculated from evaporation and potential evaporaiton'%args.model.upper()

                # Create Dimensions
                T, I, J = sesr.shape
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                nc.createDimension('time', size = T)

                # Create the lat, lon, and time information
                nc.createVariable('lat', lat.dtype, ('x', 'y'))
                nc.createVariable('lon', lon.dtype, ('x', 'y'))

                nc.createVariable('date', str, ('time',))

                nc.variables['lat'][:] = lat[:]
                nc.variables['lon'][:] = lon[:]
                nc.variables['date'][:] = time[:]

                # Create and save SESR
                nc.createVariable('sesr', sesr.dtype, ('time', 'x', 'y'))
                nc.variables['sesr'][:] = sesr[:]

    if args.calculate_thresholds:
        if args.level > 4:
            # Load SESR
            print('Loading SESR')
            key = None

            # Initialize lists
            variable = []
            for year in all_years:
                # Load SESR data
                with Dataset('%s/%s/fd_indices/africa_sesr_%04d.nc'%(base_path, args.model, year), 'r') as nc:
                    tmp = nc.variables['sesr'][:]

                    # Also load lat and lon information here
                    lat = nc.variables['lat'][:]
                    lon = nc.variables['lon'][:]

                    # Add ET data to the list
                    variable.append(tmp)
            
            # Convert to arrays
            variable = np.concatenate(variable)
        else:
            # SM is loaded separately
            key = 'rz' if args.level == 0 else args.level

            if args.model == 'era5':
                variable = (7/28) * sm[1] + (21/28) * sm[2] if args.level == 0 else sm[key]
            else:
                variable = (10/40) * sm[1] + (30/40) * sm[2] if args.level == 0 else sm[key]

        # Calculate the thresholds
        print('Calculating percentiles')
        thresholds, one_year = calculate_percentile_thresholds(variable, 
                                                               40, 
                                                               dates, 
                                                               datetime(1990, 1, 1),
                                                               datetime(2020, 12, 31),
                                                               days_per_year = 366)

        # Save the threshold values

        # Convert timestamps to a string for saving in .nc file
        time = np.array([date.isoformat() for date in one_year])

        # Save the results
        var_name = 'sesr' if args.level > 4 else 'swvl%s'%key
        with Dataset('%s/%s/africa_%s_40_percent_thresh.nc'%(base_path, args.model, var_name), 'w', format = 'NETCDF4') as nc:
        # with Dataset('../era5/sesr_%04d.nc'%(year), 'w', format = 'NETCDF4') as nc:
            nc.description = 'Daily %s reanalysis data for the 40th percentile value of %s over Africa'%(args.model.upper(), var_name)

            # Create Dimensions
            T, I, J = thresholds.shape
            nc.createDimension('x', size = I)
            nc.createDimension('y', size = J)
            nc.createDimension('time', size = T)

            # Create the lat, lon, and time information
            nc.createVariable('lat', lat.dtype, ('x', 'y'))
            nc.createVariable('lon', lon.dtype, ('x', 'y'))

            nc.createVariable('date', str, ('time',))

            nc.variables['lat'][:] = lat[:]
            nc.variables['lon'][:] = lon[:]
            nc.variables['date'][:] = time[:]

            # Create and save SESR
            nc.createVariable('thresholds', thresholds.dtype, ('time', 'x', 'y'))
            nc.variables['thresholds'][:] = thresholds[:]
        
    # Perform FD calculations according to evaporative stress and save the results
    if args.calculate_fd_sesr:
        # Load SESR
        print('Loading SESR')

        # Initialize lists
        sesr = []
        for year in all_years:
            # Load ET data
            with Dataset('%s/%s/fd_indices/africa_sesr_%04d.nc'%(base_path, args.model, year), 'r') as nc:
                tmp = nc.variables['sesr'][:]

                # Also load lat and lon information here
                lat = nc.variables['lat'][:]
                lon = nc.variables['lon'][:]

                # Add ET data to the list
                sesr.append(tmp)
          
        # Convert to arrays
        sesr = np.concatenate(sesr)
        
        # Calculate FD via evaporative stress
        print('Calculating FD')
        fd = christian_fd(sesr, 
                          mask, 
                          dates, 
                          start_year = 1990, 
                          end_year = 2020, 
                          apply_runmean = True)
        
        # Save FD results for for each year
        for year in all_years:
            # Isolate 1 year at a time (data saved as yearly files)
            ind = np.where(year == years)[0]

            fd_year = fd[ind,:,:]
            dates_year = dates[ind]

            # Convert timestamps to a string for saving in .nc file
            time = np.array([date.isoformat() for date in dates_year])

            # Save the results
            with Dataset('%s/%s/africa_fd_sesr_%04d.nc'%(base_path, args.model, year), 'w', format = 'NETCDF4') as nc:
                nc.description = 'Daily %s reanalysis data for identified FD according to the Christian et al. 2023 method over Africa'%args.model.upper()

                # Create Dimensions
                T, I, J = fd_year.shape
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                nc.createDimension('time', size = T)

                # Create the lat, lon, and time information
                nc.createVariable('lat', lat.dtype, ('x', 'y'))
                nc.createVariable('lon', lon.dtype, ('x', 'y'))

                nc.createVariable('date', str, ('time',))

                nc.variables['lat'][:] = lat[:]
                nc.variables['lon'][:] = lon[:]
                nc.variables['date'][:] = time[:]

                # Create and save the FDII and its components
                nc.createVariable('fd', fd_year.dtype, ('time', 'x', 'y'))
                nc.variables['fd'][:] = fd_year[:]

    # Calculate SM percentiles and save the results
    if args.calculate_sm_percentiles:
        # Create one layer for 0 - 28 cm (root zone SM)
        if args.model == 'era5':
            sm['rz'] = (7/28) * sm[1] + (21/28) * sm[2] # Weighted average based on depth of each soil layer
        else:
            sm['rz'] = (10/40) * sm[1] + (30/40) * sm[2] # Weighted average based on depth of each soil layer

        keys = [1, 2, 3, 4, 'rz']
        sm_year = {}

        # Calculate SM percentiles for each year and save the results
        for year in all_years:
            print(year)
            # Perform calculations for 1 year at a time (data saved as yearly files)
            ind = np.where(year == years)[0]

            sm_year[1] = sm[1][ind,:,:]; sm_year[2] = sm[2][ind,:,:]; sm_year[3] = sm[3][ind,:,:]; sm_year[4] = sm[4][ind,:,:]; sm_year['rz'] = sm['rz'][ind,:,:]
            dates_year = dates[ind]

            # Convert timestamps to a string for saving in .nc file
            time = np.array([date.isoformat() for date in dates_year])
            
            # Calculate the percentiles for each layer
            for key in keys:
                # Skip the calculations if the file already exists
                if os.path.exists('%s/%s/africa_soil_moisture_percentiles_%s_%04d.nc'%(base_path, args.model, str(key), year)):
                    continue

                # Calculate the SM percentiles
                smp = calculate_sm_percentiles(sm_year[key], 
                                               sm[key], 
                                               dates_year, 
                                               dates, 
                                               mask = mask)
                
                # Save the results
                with Dataset('%s/%s/africa_soil_moisture_percentiles_%s_%04d.nc'%(base_path, args.model, str(key), year), 'w', format = 'NETCDF4') as nc:
                    # Determine the depth of the soil layer
                    if key == 1:
                        depth = '0 - 7' if args.model == 'era5' else '0 - 10'
                    elif key == 2:
                        depth = '7 - 28' if args.model == 'era5' else '10 - 40'
                    elif key == 3:
                        depth = '28 - 100' if args.model == 'era5' else '40 - 100'
                    elif key == 4:
                        depth = '100 - 289' if args.model == 'era5' else '100 - 200'
                    else: # All that remains here is the root zone layer
                        depth = '0 - 28' if args.model == 'ear5' else '0 - 40'

                    nc.description = 'Daily %s reanalysis data for %s cm soil moisture percentiles over Africa'%(args.model.upper(), depth)

                    # Create Dimensions
                    T, I, J = smp.shape
                    nc.createDimension('x', size = I)
                    nc.createDimension('y', size = J)
                    nc.createDimension('time', size = T)

                    # Create the lat, lon, and time information
                    nc.createVariable('lat', lat.dtype, ('x', 'y'))
                    nc.createVariable('lon', lon.dtype, ('x', 'y'))

                    nc.createVariable('date', str, ('time',))

                    nc.variables['lat'][:] = lat[:]
                    nc.variables['lon'][:] = lon[:]
                    nc.variables['date'][:] = time[:]

                    # Create and save the percentiles
                    nc.createVariable('smp%s'%str(key), smp.dtype, ('time', 'x', 'y'))
                    nc.variables['smp%s'%str(key)][:] = smp[:]

    # Calculate the FDII for each layer
    if args.calculate_fdii:
        #keys = [1, 2, 3, 4, 'rz']
        key = 'rz' if args.level == 0 else args.level

        # # Calculate the FDII for each layer
        # for key in keys:
        # Calculate the FDII and its components
        fdii, fd_int, dro_sev = calculate_fdii(smp[key],
                                               dates, 
                                               apply_runmean = True,
                                               mask = mask)

        # Save the FDII results for each year
        for year in all_years:
            # Isolate 1 year at a time (data saved as yearly files)
            ind = np.where(year == years)[0]

            fdii_year = fdii[ind,:,:]; fd_int_year = fd_int[ind,:,:]; dro_sev_year = dro_sev[ind,:,:]
            dates_year = dates[ind]

            # Convert timestamps to a string for saving in .nc file
            time = np.array([date.isoformat() for date in dates_year])

            # Save the results
            with Dataset('%s/%s/africa_fdii_%s_%04d.nc'%(base_path, args.model, str(key), year), 'w', format = 'NETCDF4') as nc:
                # Determine the depth of the soil layer
                if key == 1:
                    depth = '0 - 7' if args.model == 'era5' else '0 - 10'
                elif key == 2:
                    depth = '7 - 28' if args.model == 'era5' else '10 - 40'
                elif key == 3:
                    depth = '28 - 100' if args.model == 'era5' else '40 - 100'
                elif key == 4:
                    depth = '100 - 289' if args.model == 'era5' else '100 - 200'
                else: # All that remains here is the root zone layer
                    depth = '0 - 28' if args.model == 'era5' else '0 - 40'

                nc.description = 'Daily %s reanalysis data for %s cm Flash Drought Intensity Index over Africa'%(args.model.upper(), depth)

                # Create Dimensions
                T, I, J = fdii_year.shape
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                nc.createDimension('time', size = T)

                # Create the lat, lon, and time information
                nc.createVariable('lat', lat.dtype, ('x', 'y'))
                nc.createVariable('lon', lon.dtype, ('x', 'y'))

                nc.createVariable('date', str, ('time',))

                nc.variables['lat'][:] = lat[:]
                nc.variables['lon'][:] = lon[:]
                nc.variables['date'][:] = time[:]

                # Create and save the FDII and its components
                nc.createVariable('fd_int%s'%str(key), fd_int_year.dtype, ('time', 'x', 'y'))
                nc.variables['fd_int%s'%str(key)][:] = fd_int_year[:]

                nc.createVariable('dro_sev%s'%str(key), dro_sev_year.dtype, ('time', 'x', 'y'))
                nc.variables['dro_sev%s'%str(key)][:] = dro_sev_year[:]

                nc.createVariable('fdii%s'%str(key), fdii_year.dtype, ('time', 'x', 'y'))
                nc.variables['fdii%s'%str(key)][:] = fdii_year[:]

    # Calculate Yuan et al FD and save the results
    if args.calculate_fd_yuan:
        # keys = [1, 2, 3, 4, 'rz']
        key = 'rz' if args.level == 0 else args.level

        # # identify the FD for each layer
        # for key in keys:
        # Calculate the FDII and its components
        fd = yuan_fd(smp[key],
                     mask, 
                     dates, 
                     apply_runmean = True)

        # Save the FDII results for each year
        for year in all_years:
            # Isolate 1 year at a time (data saved as yearly files)
            ind = np.where(year == years)[0]

            fd_year = fd[ind,:,:]
            dates_year = dates[ind]

            # Convert timestamps to a string for saving in .nc file
            time = np.array([date.isoformat() for date in dates_year])

            # Save the results
            with Dataset('%s/%s/africa_fd_sm_%s_%04d.nc'%(base_path, args.model, str(key), year), 'w', format = 'NETCDF4') as nc:
                # Determine the depth of the soil layer
                if key == 1:
                    depth = '0 - 7' if args.model == 'era5' else '0 - 10'
                elif key == 2:
                    depth = '7 - 28' if args.model == 'era5' else '10 - 40'
                elif key == 3:
                    depth = '28 - 100' if args.model == 'era5' else '40 - 100'
                elif key == 4:
                    depth = '100 - 289' if args.model == 'era5' else '100 - 200'
                else: # All that remains here is the root zone layer
                    depth = '0 - 28' if args.model == 'era5' else '0 - 40'

                nc.description = 'Daily %s reanalysis data for identified FD according to the Yuan et al. 2023 method over Africa for the %s cm soil layer'%(args.model.upper(), depth)

                # Create Dimensions
                T, I, J = fd_year.shape
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                nc.createDimension('time', size = T)

                # Create the lat, lon, and time information
                nc.createVariable('lat', lat.dtype, ('x', 'y'))
                nc.createVariable('lon', lon.dtype, ('x', 'y'))

                nc.createVariable('date', str, ('time',))

                nc.variables['lat'][:] = lat[:]
                nc.variables['lon'][:] = lon[:]
                nc.variables['date'][:] = time[:]

                # Create and save the FDII and its components
                nc.createVariable('fd%s'%str(key), fd_year.dtype, ('time', 'x', 'y'))
                nc.variables['fd%s'%str(key)][:] = fd_year[:]

