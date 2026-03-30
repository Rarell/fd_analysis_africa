import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

# All acceptable subsets and their upper and low lat/lon
subsets = { # Formatted as lower_lat, upper_lat, lower_lon, upper_lon
    'nh': [23.5, 90, 0, 360],
    'sh': [-90, -23.5, 0, 360],
    'tropics': [-23.5, 23.5, 0, 360],
    'africa': [-35, 35, 335, 53],
    'africa_nh': [23.5, 35, 335, 53],
    'africa_sh': [-35, -23.5, 335, 53],
    'africa_tropics': [-23.5, 23.5, 335, 53],
    'conus': [24, 55, 230, 300]
}

# List of different variable times (e.g., upper air variables, whether they have understcores in the name or whether to skip them)
upper_air_variables = ['u', 'v', 'z', 'q']
skip_variables = ['time', 'latitude', 'longitude', 'forecast_step', 'datetime']
underscore_variables = ['tp']

def subset_data(
        data, 
        latitude, 
        longitude, 
        subset
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Subset global dataset to a specific region
    Note this function takes out a box region from a larger set of spatial data

    Inputs:
    :param data: Data to make the subset for (must be 2D or 3D array)
    :param latitude: 1D list/array of laitudes for data
    :param longitude: 1D list/array of longitudes for data
    :param subset: Name of the subset region. Must be a key in subsets

    Outputs:
    :param data_sub: The subsetted dataset
    :param lat_sub: Latitudes for subsetted region
    :param lon_sub: Longitudes for subsetted region
    '''

    # Limits for specific regions
    if subset in subsets.keys():
        lower_lat = subsets[subset][0]
        upper_lat = subsets[subset][1]
        lower_lon = subsets[subset][2]
        upper_lon = subsets[subset][3]
    else:
        sub_list = [sub for sub in subsets.keys()]
        assert "subset must be one of {sub_list}, but got {subset} instead".format(sub_list = sub_list, subset = subset)
        
    # Get longitude indices for the subset
    if lower_lon > upper_lon:
        lon_ind = np.where((longitude >= lower_lon) | (longitude <= upper_lon))[0]
    else:
        lon_ind = np.where((longitude >= lower_lon) & (longitude <= upper_lon))[0]

    # Get the latitude indices for the subset
    lat_ind = np.where((latitude >= lower_lat) & (latitude <= upper_lat))[0]

    # Subset the latitude and longitude
    lat_sub = latitude[lat_ind]
    lon_sub = longitude[lon_ind]

    # Subset the data
    if len(data.shape) < 3:
        data_sub = data[:,lon_ind]
        data_sub = data_sub[lat_ind,:]
    else:
        data_sub = data[:,:,lon_ind]
        data_sub = data_sub[:,lat_ind,:]

    return data_sub, lat_sub, lon_sub


def new_sort(files) -> np.ndarray:
    '''
    Sorts .nc with string formates of 024 .. 984 to 1008 ...
    NOTE: THIS HARD ASSUMES A FILENAME STRUCTURE AND THAT THE 3 DIGIT UNDERSCORE IS ON THE 7TH FROM THE END

    Inputs:
    :param files: List if filenames to be sorted

    Outputs:
    :param files_sorted: Sorted filenames (1D np.array)
    '''

    numbers = '0123456789'

    # Assumes filename is formated as filename_###.nc or filename_####.nc; 
    # so the -7 entry is a number for 4 digit numbers, and underscore for 3 digit
    underscore = []
    non_underscore = []

    # Parse the list of files into two sets 3 digit and 4 digit numbers
    for file in files:
        underscore.append(file) if file[-7] not in numbers else non_underscore.append(file)

    # Sets of files can now be sorted without confusion
    underscore = np.sort(underscore)
    non_underscore = np.sort(non_underscore)

    # Concatenate files together to deliver sorted filenames
    files_sorted = np.concatenate([underscore, non_underscore])

    return files_sorted



def get_metric_information(metric) -> Tuple[str, str, int]:
    '''
    Given a string from CREDIT metrics column, parse the name into separate portions

    Inputs:
    :param metric: Raw name of the metric in the CREDIT csv file

    Outputs:
    :param metric_name: The name of the metric skill score
    :param var_name: The name of the variable evaluated
    :param level: The index for the corresponding pressure level for upper air variables (None if not an upper air variable)
    '''

    # Skip variables do not have compacted names, and they are what the metrics get plotted against
    if metric in skip_variables: 
        return

    # Determine if metric is a performance metric over all variables (in which case metric only contains metric_name)
    if len(metric) < 5:
        var_name = 'Overall' # For overall metrics, only the metric name is in it
        metric_name = metric
        level = None

    # Determine if the metric is for an upper air variable
    elif metric.split('_')[1] in upper_air_variables:
        # Note q_tot gets an extra split and needs special attention to be reconstructed
        if metric.split('_')[1] == upper_air_variables[-1]: 
            metric_name, var1, var2, level = metric.split('_')

            # Special attention for q_tot
            var_name = var1 + '_' + var2 
        else:
            # Split metric into metric_name, var_name, and level
            metric_name, var_name, level = metric.split('_')

        # Make sure level is an int so it can be used as an index
        level = int(level)

    # Underscore variables (e.g., tp_7d, tp_14d, tp_30d) require special attention since they split into an extra entry
    elif (metric.split('_')[1] in underscore_variables) & (len(metric.split('_')) > 2):
        metric_name, var1, var2 = metric.split('_')

        # Special attention for underscore variables (note this is for non upper-air variables; level = None)
        var_name = var1 + '_' + var2
        level = None

    else:
        # Split non upper air variables into metric_name and var_name (level = None for non upper-air)
        metric_name, var_name = metric.split('_')
        level = None

    return metric_name, var_name, level

def least_squares(x, y) -> Tuple[float, float]:
    '''
    Perform a linear least-squares estimate on a time series (i.e., estimate y using x with a linear regression)
    
    Inputs:
    :param x,y: Input time series to be related via regression line
    
    Outputs;
    :param xhat[0]: Linear slope value of the regressed time series
    :param xhat[1]: Linear intercept value of the regressed time series
    '''
    
    # Get the length of the timeseries
    T = x.size
    
    # Initialize some variables
    t = np.arange(T)
    E = np.ones((T, 2))
    
    # Define the model matrix
    E[:,0] = x # x estimates
    E[:,1] = 1 # bias term
    
    # Use least squares (matrix form) to find the linear regression
    invEtE = np.linalg.inv(np.dot(E.T, E))
    xhat = invEtE.dot(E.T).dot(y)
    
    return xhat[0], xhat[1]

def wind_speed(u, v) -> float:
    '''
    Calculate the wind speed from loaded u and v components
    '''

    # Calculate WS
    ws = np.sqrt(u**2 + v**2)
    
    return ws

def vapor_pressure_deficit(
        temperature, 
        dewpoint, 
        convert_to_celsius: bool = True
        ) -> np.ndarray:
    '''
    Calculate vapor pressure deficit (VPD) (Pa), from the temperature (K) and dewpoint temperature (K)
    '''

    # The empirical form of the CC equation uses temperature/dewpoint in Celsius
    if convert_to_celsius:
        temperature = temperature - 273.15
        dewpoint = dewpoint - 273.15

    # Coefficients to empirical CC equation
    e0 = 611.2 # Pa
    a = 17.67
    b = 243.5

    # Use the empirical CC equation to get vapor pressure from dewpoint
    exponent = a * dewpoint/(dewpoint + b)
    e = e0 * np.exp(exponent)

    # Use the empirical CC equation to get saturation vapor pressure from temperature
    exponent = a * temperature/(temperature + b)
    e_s = e0 * np.exp(exponent)

    # Calculate VPD
    vpd = e_s - e

    return vpd

def standardize_variable(
        variable, 
        dates_all, 
        start,
        end,
        days_per_year: int = 366
        ) -> np.ndarray:
    '''
    Calculates the standardized anomalies of a daily ERA5 variable.
    Climatological data is calculated for all grid points and for all timestamps in the year.

    Inputs:
    :param variable: Dataset to be standardized (np.ndarray, with shape time x lat x lon)
    :param dates_all: Datetimes labels for each time step in e and pet (np.ndarray with shape time)
    :param start: Datetime of first date in the climatology to consider (e.g., Jan 1, 1980)
    :param end: Datetime of end date in climatology to consider (e.d., Dec 31, 2020)
    :param days_per_year: Total number of days in one year of data (use 366 if using daily data to include leap day)

    Outputs:
    :param anomalies: Standardized value of variable for each grid and date in year (np.ndarray of shape time x lat x lon)
    '''
    

    T, I, J = variable.shape
    anomalies = np.ones((T, I, J))
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

    print('Initialized variables, calculation means')

    # Conduct climatology calculations
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]

        # Sum over all all ESR in a given day
        tmp_sum = np.nansum(variable[ind,:,:], axis = 0)
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
        error = np.nansum((variable[ind,:,:] - means[t,:,:])**2, axis = 0)
        stds[t,:,:] = np.nansum([stds[t,:,:], error], axis = 0)

    # One final loop to finish standard deviation calculations
    for t, date in enumerate(dates_year):
        stds[t,:,:] = np.sqrt(stds[t,:,:]/(N[t] - 1))

    stds = stds.astype(np.float32)
    print(np.min(means), np.max(means))
    print(np.min(stds), np.max(stds))

    print('Standard deviations calculated, standardizing variable')
    days = np.array([day.day for day in dates_year])
    months = np.array([day.month for day in dates_year])

    # Calcualte the standardized anomalies
    for t, date in enumerate(dates_all):
        # Find the date index for the one year range
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        # Standardize the ESR to get SESR
        anomalies[t,:,:] = (variable[t,:,:] - means[ind[0],:,:])/stds[ind[0],:,:]

    anomalies = anomalies.astype(np.float32)

    return anomalies