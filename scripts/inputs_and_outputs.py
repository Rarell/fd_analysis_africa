'''Provides functions for loading and
saving datasets
'''

import numpy as np
import pickle
from glob import glob
from typing import Tuple
from netCDF4 import Dataset
from datetime import datetime, timedelta

from utils import wind_speed, vapor_pressure_deficit, dewpoint

# Pre-determine pathing information for the datasets
raw_data_paths = {
    'tair': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/temperature',
    'd2m': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/moisture_surface', 
    'sp': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/pressure', 
    'ws': ['/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/wind_speed', '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/wind_speed'],
    'e': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/evaporation', 
    'pev': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/potential_evaporation', 
    'tp': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/precipitation', 
    'vpd': ['/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/temperature', '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/moisture_surface'], 
    'swvl1': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/liquid_vsm', 
    'swvl2': '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/liquid_vsm', 
    'swvlrz': ['/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/liquid_vsm', '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s/liquid_vsm'],
    'enso': '/ourdisk/hpc/ai2es/sedris/climate_indices',
    'iod': 'dmi.had.long',
    'mjo': 'mjo.timeseries',
}

# Pre-determine the filename of each dataset
raw_data_base_names = {
    'era5': {
        'tair': 'africa_2m_temperature_',
        'd2m': 'africa_2m_dewpoint_', 
        'sp': 'africa_surface_pressure_', 
        'ws': ['africa_10m_u_component_of_wind_', 'africa_10m_v_component_of_wind_'], 
        'e': 'africa_evaporation_', 
        'pev': 'africa_potential_evaporation_', 
        'tp': 'africa_total_precipitation_', 
        'vpd': ['africa_2m_temperature_', 'africa_2m_dewpoint_'], 
        'swvl1': 'africa_volumetric_soil_water_layer_1_', 
        'swvl2': 'africa_volumetric_soil_water_layer_2_', 
        'swvlrz': ['africa_volumetric_soil_water_layer_1_', 'africa_volumetric_soil_water_layer_2_', 'africa_volumetric_soil_water_root_zone_']
    },
    'gldas': {
        'tair': 'africa_gldas.temperature.daily_',
        'd2m': 'africa_gldas.specific_humidity.daily_', 
        'sp': 'africa_gldas.pressure.daily_', 
        'ws': 'africa_gldas.wind_speed.daily_', 
        'e': 'africa_gldas.evaporation.daily_', 
        'pev': 'africa_gldas.potential_evaporation.daily_', 
        'tp': 'africa_gldas.precipitation.daily_', 
        'vpd': ['africa_gldas.temperature.daily_', 'africa_gldas.specific_humidity.daily_'], 
        'swvl1': 'africa_gldas.soil_moisture_0-10cm.daily_', 
        'swvl2': 'africa_gldas.soil_moisture_10-40cm.daily_', 
        'swvlrz': ['africa_gldas.soil_moisture_0-10cm.daily_', 'africa_gldas.soil_moisture_10-40cm.daily_', 'africa_gldas.soil_moisture_root_zone.daily_']
    },
    'enso': 'enso.timeseries',
    'iod': 'dmi.had.long',
    'mjo': 'mjo.timeseries'
}

# Pre-determine the short names (used in nc files) for each variable for GLDAS
gldas_snames = {
    'tair': 'temp',
    'd2m': 'q', 
    'sp': 'pres', 
    'ws': 'wspd', 
    'e': 'evap', 
    'pev': 'pevap', 
    'tp': 'precip', 
    'vpd': 'vpd', 
    'swvl1': 'soilm', 
    'swvl2': 'soilm', 
    'swvlrz': 'swvlrz'
}

# Determine which variables are climate indices
climate_indices = ['enso', 'iod', 'mjo']

def load_fd_one_year(
        file, 
        sname, 
        times: str = 'all'
        ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load 1 year of FD data

    Inputs:
    :param file: path + filename of the year of FD data being loaded
    :param sname: Dictionary key of the FD variable being loaded 
    :param times: String indicating whether to return full year of FD, or only for summer/winter months

    Outputs:
    :param fd: One year of FD data loaded (np.ndarray with shape time x lat x lon)
    :param dates_dt: Array of datetime timestamps for each time in FD (np.ndarray with shape time)
    '''

    # Load the FD data
    with Dataset(file, 'r') as nc:
        fd = nc.variables[sname][:]
        dates = nc.variables['date'][:]

        # Convert dates to datetimes
        dates_dt = np.array([datetime.fromisoformat(date) for date in dates])

        # Collect months
        months = np.array([date.month for date in dates_dt])

        # If desired, focus on "summer" months (MAMJJA)
        if times == 'summer':
            ind = np.where( (months >= 3) & (months <= 8) )[0]
            fd = fd[ind,:,:]
            dates_dt = dates_dt[ind]

        # If desired, focus on "winter" months (SONDJF)
        elif times == 'winter':
            ind = np.where( (months >= 9) | (months <= 2) )[0]
            fd = fd[ind,:,:]
            dates_dt = dates_dt[ind]
            
    return fd, dates_dt

def load_index_one_year(
        file, 
        sname, 
        index_base, 
        times: str = 'all',
        version = 'v2.1',
        I: int = 180,
        J: int = 360,
        ) -> np.ndarray:
    '''
    Load one year of index data (i.e., SESR index, FDII index, or SM)

    Inputs:
    :param file: Path + filename of the year of index data to load
    :param sname: Dictionary key of the data to load; if None no data is loaded
    :param index_base: Base filename of the index data
    :param times: String indicating whether to return full year of index data, or only for summer/winter months
    :param version: String indicating GLDAS version being loaded (for ERA5, v2.2 also denotes to load 0 - 100 cm SM data)
    :param I, J: Spatial dimensions used to make placeholder data if sname == None

    Outputs:
    :param index_data: Loaded index data
    '''
    # Load the index data
    if sname is not None:
        # For root zone, two soil layers need to be loaded to calculate RZSM
        if ('rz' in index_base) & np.invert(version == 'v2.2'):
        	# Load the top layer of SM data
            with Dataset(file[0], 'r') as nc:
                index_data_1 = nc.variables[sname[0]][:]
                dates = nc.variables['date'][:]

                # Convert dates to datetimes
                dates_dt = np.array([datetime.fromisoformat(date) for date in dates])

                # Collect months
                months = np.array([date.month for date in dates_dt])
            
            # Load the second layer of SM data
            with Dataset(file[1], 'r') as nc:
                index_data_2 = nc.variables[sname[1]][:]

            # Calculate root zone SM (RZSM)
            index_data = (7/28) * index_data_1 + (21/28) * index_data_2

        else:
            # Load index data
            with Dataset(file, 'r') as nc:
                index_data = nc.variables[sname][:]
                dates = nc.variables['date'][:]

                # Convert dates to datetimes
                dates_dt = np.array([datetime.fromisoformat(date) for date in dates])

                # Collect months
                months = np.array([date.month for date in dates_dt])

        # If desired, focus on "summer" months (MAMJJA)
        if times == 'summer':
            ind = np.where( (months >= 3) & (months <= 8) )[0]
            index_data = index_data[ind,:,:]

        # If desired, focus on "winter" months (SONDJF)
        elif times == 'winter':
            ind = np.where( (months >= 9) | (months <= 2) )[0]
            index_data = index_data[ind,:,:]
    
    else:
        # Placeholder so index_total can still be called when using FDII without 
        # significant code changes or further bloating the params argument (in analysis.py)
        index_data = np.zeros((1, I, J)) * np.nan 

    return index_data

def load_raw_data(sname, model, version = 'v2.1') -> np.ndarray:
    '''
    Load a set of raw data for a given variable
    
    Inputs:
    :param sname: Dictionary key of the data to load
    :param model: Reanalysis model the dataset comes from ("era5" or "gldas")
    :param version: GLDAS version to reference (and whether to reference root_zone layer in ERA5 soil moisture)
    
    Outputs:
    :param data: Raw data loaded for all years available (np.ndarray with shape time x lat x lon)
    '''

    # Climate indices are located elsewhere, and requires different processing
    if sname in climate_indices:
        # Load the climate index, and make it a np.ndarray, full timeseries
        data = load_climate_index(sname, model)
        return data

    # Collect the path to the variable
    path = raw_data_paths[sname]
    # Note some variables require multiple paths because they are calculated from multiple variables
    if isinstance(path, list) & (sname != 'vpd') & (sname != 'ws'):
        path = [p%model for p in path]
    elif ((sname == 'ws') & (model == 'era5')) | (sname == 'vpd'):
        path = [p%model for p in path]
    elif (model == 'gldas') & (sname == 'ws'):
        path = path[0]%model
    else:
        path = path%model

    # Collect the base names of the variable
    base_fn = raw_data_base_names[model][sname] if np.invert(version == 'v2.2') else raw_data_base_names[model][sname][-1]

	# GLDAS has different keys from what will be given; collect the correct one
    if model == 'gldas':
        sname = gldas_snames[sname]

    # Collect the base filenames of the variable
    if isinstance(base_fn, list):
        # Collect multiple set of filenames if multiple variables are needed
        files = glob('%s/%s*.nc'%(path[0], base_fn[0]), recursive = True)
        files_2 = glob('%s/%s*.nc'%(path[1], base_fn[1]), recursive = True)
        files_2 = np.sort(files_2)
    else:
        files = glob('%s/%s*.nc'%(path, base_fn), recursive = True)

    data = []

	# Load data for each file (one file = one year of data) at a time, 
	# concatenate into one array at the end 
    for n, file in enumerate(np.sort(files)):
        if sname == 'ws':
            # Load u and v components
            with Dataset(file, 'r') as nc:
                u = nc.variables['u10'][:]
            
            with Dataset(files_2[n], 'r') as nc:
                v = nc.variables['v10'][:]

            # Calculate wind speed
            ws = wind_speed(u, v)
            data.append(ws)
            
        elif sname == 'vpd':
            if model == 'era5':
                # Load T and T_d
                with Dataset(file, 'r') as nc:
                    if model == 'era5':
                        # Some funkiness with ERA5 dictionary keys for the temperature data
                        keys = nc.variables.keys()
                        sname_t = 'tair' if 'tair' in keys else 't2m'
                    else:
                        sname_t = sname
                    tair = nc.variables[sname_t][:]
                
                with Dataset(files_2[n], 'r') as nc:
                    tdew = nc.variables['d2m'][:]
            elif model == 'gldas':
                # Load T and q
                with Dataset(file, 'r') as nc:
                    tair = nc.variables['temp'][:]
                
                with Dataset(files_2[n], 'r') as nc:
                    q = nc.variables['q'][:]

                # Also load p
                path_p = raw_data_paths['sp']%model
                base_fn_p = raw_data_base_names[model]['sp']
                files_3 = glob('%s/%s*.nc'%(path_p, base_fn_p), recursive = True)
                with Dataset(np.sort(files_3)[n], 'r') as nc:
                    p = nc.variables['pres'][:]

                # Calculate dewpoint from q and p
                tdew = dewpoint(q, p)

            # Calculate vapor pressure deficit from T and T_d
            vpd = vapor_pressure_deficit(tair, tdew)
            data.append(vpd)
        elif sname == 'swvlrz':
            # Version 2.2 of GLDAS has RZSM already calculated (one of only three SM options)
            if version == 'v2.2':
                with Dataset(file, 'r') as nc:
                    rzsm = nc.variables['swvlrz'][:] if model == 'era5' else nc.variables['soilm'][:]
            else:
                # Load soil moisture for the two root zone layers
                with Dataset(file, 'r') as nc:
                    sm1 = nc.variables['swvl1'][:] if model == 'era5' else nc.variables['soilm'][:]
                
                with Dataset(files_2[n], 'r') as nc:
                    sm2 = nc.variables['swvl2'][:] if model == 'era5' else nc.variables['soilm'][:]

                # Calculate RZSM
                rzsm = (7/28) * sm1 + (21/28) * sm2 if model == 'era5' else (10/40) * sm1 + (30/40) * sm2

            data.append(rzsm)
        elif (sname == 'q') & (model == 'gldas'):
            # Load q
            with Dataset(file, 'r') as nc:
                q = nc.variables['q'][:]

            # Also load p
            path_p = raw_data_paths['sp']%model
            base_fn_p = raw_data_base_names[model]['sp']
            files_3 = glob('%s/%s*.nc'%(path_p, base_fn_p), recursive = True)
            with Dataset(np.sort(files_3)[n], 'r') as nc:
                p = nc.variables['pres'][:]

            # Calculate dewpoint from q and p
            tdew = dewpoint(q, p)

            data.append(tdew)
        elif sname == 'tair':
            # Load the data; note with T, some snames may be t2m instead of tair
            # i.e., some funkiness with ERA5 dictionary keys for the temperature data
            with Dataset(file, 'r') as nc:
                keys = nc.variables.keys()
                sname_new = 'tair' if 'tair' in keys else 't2m'
                data.append(nc.variables[sname_new][:])
        else:
            # Load the data
            with Dataset(file, 'r') as nc:
                data.append(nc.variables[sname][:])

    # Turn the data into an array
    data = np.concatenate(data, axis = 0)

    return data

def load_climate_index(sname, model: str = 'era5') -> np.ndarray:
    '''
    Load a climate index and process it into a daily gridded array
    
    At each grid point the climate index will repeat, but can be compared with other
    gridded datasets
    
    Inputs:
    :param sname: Dictionary key of the climate index to load
    :param model: Reanalysis model the dataset to generate the grid for ("era5" or "gldas")
    
    Outputs:
    :param data: Raw data loaded for all years available (np.ndarray with shape time x lat x lon)
    '''
    
    # Load the mask as a reference grid
    with Dataset('%s/%s/aridity_mask.nc'%('/ourdisk/hpc/ai2es/sedris/fd_analysis/data', model), 'r') as nc:
        mask = nc.variables['aim'][0,:,:]

    # Hard fix the path to the climate indices
    index_path = '/ourdisk/hpc/ai2es/sedris/climate_indices' 

    # Determine if ENSO is being loaded (its csv has multiple indices to load for SSTs, SSTAs, and for each ENSO region)
    usecols = (1, 2, 3, 4, 5, 6, 7, 8) if sname.lower() == 'enso' else 1

	# MJO also has multiple entries; take the third column (15 day running average applied to MJO)
    if sname.lower() == 'mjo':
        usecols = 2

    # Determine the filename
    filename = raw_data_base_names[sname]

    # Note IOD and MJO data need their time stamps prepared
    timestamps_prepared = False if (sname.lower() == 'iod') | (sname.lower() == 'mjo') else True

    # Load the data
    data = np.loadtxt('%s/%s.csv'%(index_path, filename), delimiter = ',', skiprows = 1, usecols = usecols)

    # Load timestamps and turn them into datetimes arrays
    timestamps = np.loadtxt('%s/%s.csv'%(index_path, filename), delimiter = ',', dtype = str, skiprows = 1, usecols = 0)
    if timestamps_prepared:
        dates = np.array([datetime.fromisoformat(date) for date in timestamps])
    elif sname == 'mjo':
        # MJO timestamps have a unique format
        dates = np.array([datetime.strptime(date, '%m/%d/%Y') for date in timestamps])
    else:
        dates = np.array([datetime.strptime(date, '%Y-%m-%d') for date in timestamps])

    # If ENSO is the index, get the 3.4 region SSTAs
    if sname.lower() == 'enso':
        data = data[:,5]

    # Construct a set of daily timestamps
    start = datetime(1979, 1, 1); end = datetime(2024, 12, 31)
    Ndays = (end - start).days
    dates_new = np.array([start + timedelta(days = day) for day in range(Ndays+1)])

    # Interpolate to a daily time series to match the other datasets
    T = dates_new.size
    index_new_dates = np.ones((T,)) * np.nan
    for t, date in enumerate(dates_new):
        # Find the most recent time stamp from grid_old
        ind = np.where(date >= dates)[0]

        # If no most recent dates were found (ENSO data starts in 1981), skip
        if len(ind) < 1:
            continue
        else:
            ind = ind[-1]
            # Use the most recent time stamp for the current, finer time resolution
            index_new_dates[t] = data[ind]

    # Interpolate to a gridded data to match the other datasets
    I, J = mask.shape
    index_data = np.ones((T, I, J))
    for i in range(I):
        for j in range(J):
            index_data[:,i,j] = index_new_dates

    return index_data

def save_pickle(file, data, snames) -> None:
    '''
    Save several datasets to a pickle file

    Inputs:
    :param file: Path + filename of the pickle file to create
    :param data: List of datasets to save
    :param snames: List of dictionary keys for each dataset in data
    '''

    # Convert the data to a dictionary to all be saved at once
    data_dict = {}
    for n in range(len(data)):
        data_dict[snames[n]] = data[n]

    # Save the dataset
    with open(file, 'wb') as f:
        pickle.dump(data_dict, f)

def load_pickle(file) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load a pickle file.

    Note this is set to specifically load FD characteristics of frequency, duration, and seveirty, 
    and assumes the data is saved in the pickle file as a dictionary

    Inputs:
    :param file: Path + filename of the pickle file

    Outputs:
    :param frequency: Fd frequency for every grid and year (np.ndarray with shape time x lat x lon)
    :param duration: Fd duration for every grid and year (np.ndarray with shape time x lat x lon)
    :param severity: Fd severity for every grid and year (np.ndarray with shape time x lat x lon)
    '''
    # Load the pickle file
    with open(file, 'rb') as f:
        data = pickle.load(f)
        frequency = data['freq']
        duration = data['dur']
        severity = data['sev']

    return frequency, duration, severity

