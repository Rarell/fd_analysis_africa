'''Prepare raw datasets by loading them from 
their original GLDAS/ERA5 folders, subsetting them
to Africa, early preprocessing, and saving them in 
the local repository for this project
'''

import numpy as np
from glob import glob
from netCDF4 import Dataset

from utils import subset_data, new_sort

if __name__ == '__main__':
    # Declare all years in the dataset
    years = np.arange(1979, 2024+1)

    model = 'gldas' # or 'era5'

    # Declare all variables to be examined
    variables = {'era5': [
            '2m_temperature',
            '2m_dewpoint', 
            'total_precipitation', 
            'evaporation', 
            'potential_evaporation',
            'surface_pressure', 
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'volumetric_soil_water_layer_1',
            'volumetric_soil_water_layer_2',
            'volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4'
        ],
        'gldas':[
            'gldas.temperature.daily', 
            'gldas.precipitation.daily',
            'gldas.evaporation.daily',
            'gldas.potential_evaporation.daily',
            'gldas.pressure.daily',
            'gldas.specific_humidity.daily', # Note GLDAS2 does not have dewpoint temperature; this will be calculated from q
            'gldas.wind_speed.daily',
            'gldas.soil_moisture_0-10cm.daily',
            'gldas.soil_moisture_10-40cm.daily',
            'gldas.soil_moisture_40-100cm.daily',
            'gldas.soil_moisture_100-200cm.daily'
        ]
        }

    # Name of the directory each corresponding variable is located in
    directories = {'era5': [
            'temperature',
            'moisture_surface', 
            'precipitation', 
            'evaporation', 
            'potential_evaporation', 
            'pressure',
            'wind_speed',
            'wind_speed',
            'liquid_vsm',
            'liquid_vsm',
            'liquid_vsm',
            'liquid_vsm'
    ],
    'gldas': [
        'temperature',
        'precipitation',
        'evaporation',
        'potential_evaporation',
        'pressure',
        'surface_moisture',
        'wind_speed',
        'soil_moisture_0-10cm',
        'soil_moisture_10-40cm',
        'soil_moisture_40-100cm',
        'soil_moisture_100-200cm'
    ]
    }

    # Base paths
    base_path = '/ourdisk/hpc/ai2es/sedris/%s'%model
    base_write = '/ourdisk/hpc/ai2es/sedris/fd_analysis/data/%s'%model

    for m, variable in enumerate(variables[model]):
        # Collect all the .nc files to be examined
        files = glob('%s/%s/%s*.nc'%(base_path, directories[model][m], variable), recursive = True)

        # Process each file
        for n, file in enumerate(new_sort(files)):
            print(file)
            # Load the data
            with Dataset(file, 'r') as nc:
                lat = nc.variables['lat'][:]
                lon = nc.variables['lon'][:]
                dates = nc.variables['date'][:]

                desc = nc.description

                for key in nc.variables.keys():
                    # Load only the main data here and save the short name
                    if np.invert(key == 'lat') & np.invert(key == 'lon') & np.invert(key == 'date'):
                        data = nc.variables[key][:]
                        data_key = key

            # Subset the data
            if model == 'era5':
                lat = lat[:,0]; lon = lon[0,:]
            elif model == 'gldas':
                # For GLDAS, convert longitude to the same format as ERA5
                lon = np.where(lon < 0, lon + 360, lon)

            data_sub, lat_sub, lon_sub = subset_data(data, lat, lon, subset = 'africa')
            T, I, J = data_sub.shape

            if model == 'gldas':
                # Remove points labeled as bad data in GLDAS
                data_sub = np.where(data_sub < -900, np.nan, data_sub)

            # Make the lat and lon 2D
            lon_sub, lat_sub = np.meshgrid(lon_sub, lat_sub)

            # Write the data
            with Dataset('%s/africa_%s_%04d.nc'%(base_write, variable, years[n]), 'w', format = 'NETCDF4') as nc:
                # Write a description for the .nc file
                nc.description = desc

                # Create the spatial and temporal dimensions
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                nc.createDimension('time', size = T)
                
                # Create the lat and lon variables       
                nc.createVariable('lat', lat_sub.dtype, ('x', 'y'))
                nc.createVariable('lon', lon_sub.dtype, ('x', 'y'))
                
                nc.variables['lat'][:,:] = lat_sub[:,:]
                nc.variables['lon'][:,:] = lon_sub[:,:]
                
                # Create the date variable
                nc.createVariable('date', str, ('time', ))
                for n in range(len(dates)):
                    nc.variables['date'][n] = str(dates[n])
                    
                # Create the main variable
                nc.createVariable(data_key, data_sub.dtype, ('time', 'x', 'y'))
                nc.variables[str(data_key)][:,:,:] = data_sub[:,:,:]

    