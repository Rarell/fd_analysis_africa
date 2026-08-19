import os, sys, warnings
import gc
import numpy as np
from typing import Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
from netCDF4 import Dataset
from argparse import ArgumentParser
from scipy import interpolate
from scipy import signal
from tqdm import tqdm

if __name__ == '__main__':
    model = 'gldas' # or era5

    days_per_year = 365

    # Define the base path to the datasets
    base_path = '/ourdisk/hpc/ai2es/sedris/fd_analysis/data'

    # Determine all years and build datetimes
    # all_years = np.arange(1979, 2024+1)
    all_years = np.arange(1979, 2003+1)

    start = datetime(1979, 1, 1)
    N_leap_days = np.sum((all_years % 4) == 0) # Note datetimes follow the scheme of a leap day every 4 years, even if it isn't exactly correct
    T_full = (days_per_year * all_years.size) + N_leap_days
    dates = np.array([start + timedelta(days = t) for t in range(T_full)])

    # Collect an array of all year values
    years = np.array([date.year for date in dates])

    # Determine the filenames of the SM datasets
    if model == 'era5':
        base_sm1_fn = 'africa_volumetric_soil_water_layer_1'
        base_sm2_fn = 'africa_volumetric_soil_water_layer_2'
        base_sm3_fn = 'africa_volumetric_soil_water_layer_3'
        sname1 = 'swvl1'; sname2 = 'swvl2'; sname3 = 'swvl3'
    else:
        base_sm1_fn = 'africa_gldas.soil_moisture_0-10cm.daily'
        base_sm2_fn = 'africa_gldas.soil_moisture_10-40cm.daily'
        base_sm3_fn = 'africa_gldas.soil_moisture_40-100cm.daily'
        sname1 = sname2 = sname3 = 'soilm'

    # Initialize lists
    sm = {}
    sm[1] = []; sm[2] = []; sm[3] = []

    if model == 'gldas':
        model_path = 'gldas/v2.2'
    else:
        model_path = 'era5'

    for year in all_years:
        print(year)       
        # Load the top layer of SM data
        with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, model_path, base_sm1_fn, year), 'r') as nc:
            tmp = nc.variables[sname1][:]

            # Also load lat and lon information here
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]

            sm[1].append(tmp)

        # Load the second layer of the SM data
        with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, model_path, base_sm2_fn, year), 'r') as nc:
            tmp = nc.variables[sname2][:]

            sm[2].append(tmp)

        # Load the third layer of SM data
        with Dataset('%s/%s/liquid_vsm/%s_%04d.nc'%(base_path, model_path, base_sm3_fn, year), 'r') as nc:
            tmp = nc.variables[sname3][:]

            sm[3].append(tmp)

    # Convert to arrays
    sm[1] = np.concatenate(sm[1]); sm[2] = np.concatenate(sm[2]); sm[3] = np.concatenate(sm[3])

    # Determine RZSM
    if model == 'era5':
        sm['rz'] = (7/100) * sm[1] + (21/100) * sm[2] + (72/100) * sm[3] # Weighted average based on depth of each soil layer
    else:
        sm['rz'] = sm[1] + sm[2] + sm[3]# (10/100) * sm[1] + (30/100) * sm[2] + (60/100) * sm[3] # Weighted average based on depth of each soil layer

    # Save the new soil moisture calculations
    for year in all_years:
        # Isolate 1 year at a time (data saved as yearly files)
        ind = np.where(year == years)[0]

        sm_year = sm['rz'][ind,:,:]
        dates_year = dates[ind]

        # Convert timestamps to a string for saving in .nc file
        time = np.array([date.isoformat() for date in dates_year])

        # Save the results
        fn = 'africa_gldas.soil_moisture_root_zone.daily_%04d.nc'%year if model == 'gldas' else 'africa_volumetric_soil_water_root_zone_%04d.nc'%year
        with Dataset('%s/%s/%s'%(base_path, model_path, fn), 'w', format = 'NETCDF4') as nc:
            nc.description = 'Daily %s reanalysis data for root zone soil moisture (RZSM), as the 0 - 100 cm weighted average, over Africa'%model.upper()

            # Create Dimensions
            T, I, J = sm_year.shape
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

            # Create and save the RZSM and its components
            sname = 'soilm' if model == 'gldas' else 'swvlrz'
            nc.createVariable(sname, sm_year.dtype, ('time', 'x', 'y'))
            nc.variables[sname][:] = sm_year[:]
