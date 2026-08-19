'''Provide functions for statistical calculations
including regression, correlation, and significance testing
'''

import numpy as np
from typing import Tuple, Union
from tqdm import tqdm
from scipy import stats

def least_squares(x, y) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[float, float, np.ndarray]]:
    '''
    Perform a linear least-squares estimate on a time series (i.e., estimate y using x with a linear regression)
    
    Inputs:
    :param x,y: Input time series to be related via regression line
    
    Outputs:
    if x is multidimensional (multiregression):
        :param xhat: Slopes of the multigression (xhat[0,:] are slopes for x[0,:], etc.)
    if y is multidimensional (regression against multiple variables):
        :param xhat[0,:]: Array of slopes of x against y
        :param xhat[1,:]: Array of intercepts of x against y
        :param yhat: Regression estimates of y
    else:
        :param xhat[0]: Linear slope value of the regressed time series
        :param xhat[1]: Linear intercept value of the regressed time series
        :param yhat: Regression estimates of y
    '''
    # Remove NaNs
    y_filled = np.where(np.isnan(y), 0, y)
    
    # Get the length of the timeseries
    T = x.size
    
    # Initialize some variables
    if len(x.shape) > 1:
        # If x is multidimensional, it should contain multiple variables to regress
        E = x
    else:
        # Else regress a 1D x (time series) against y
        t = np.arange(T)
        E = np.ones((T, 2))
    
        # Define the model matrix
        E[:,0] = x # x estimates
        E[:,1] = 1 # bias term
    
    # Use least squares (matrix form) to find the linear regression
    invEtE = np.linalg.inv(np.dot(E.T, E))
    xhat = invEtE.dot(E.T).dot(y_filled)

    yhat = np.dot(E, xhat)
    
    if len(x.shape) > 1:
        # Return all xhat (slopes) for multiregress
        return xhat
    if len(y.shape) > 1:
        # For a single time series regression, return the slope, intercept, and estimates
        return xhat[0,:], xhat[1,:], yhat
    else:
        # Else, return the singular slope, intercept, and estimates of y
        return xhat[0], xhat[1], yhat
    

def correlate(x, y) -> Union[float, np.ndarray]:
    '''
    Calculate the Pearson correlation cofficient or anomaly correlation coefficient in time

    Inputs:
    :param x,y: Input variables to be correlated

    Outputs:
    :param r: Correlation between x & y (same type as x/y)
    '''

    # Assumes to correlate in time (axis 0)
    xy_axis = None
    if len(x.shape) > 1:
        x_axis = 0
        xy_axis = 0
    else:
        x_axis = None

    if len(y.shape) > 1:
        y_axis = 0
        xy_axis = 0
    else:
        y_axis = None

    # Avoids division by 0
    epsilon = 1e-7

    # Collect deviations from the overall mean
    xprime = x - np.nanmean(x, axis = x_axis)
    yprime = y - np.nanmean(y, axis = y_axis) 

    # Determine the denominator of the correlation separately (due to its lengthy nature)
    denominator = np.sqrt(np.nansum(xprime**2, axis = x_axis) * np.nansum(yprime**2, axis = y_axis)) + epsilon

    # Calculate the Pearson correlation coefficient
    r = np.nansum(xprime * yprime, axis = xy_axis) / denominator

    return r
    
def monte_carlo_significance(
        x, 
        y, 
        original_statistic, 
        N: int = 5000, 
        statistic: str = 'regression'
        ) -> Union[float, np.ndarray]:
    '''
    Tests the significance of the regression slope or correlation vio Monte-Carlo Bootstrapping

    Inputs:
    :param x, y: Two variables being compared via a statistic
    :param original_statistic: Value of the statistic being tested
    :param N: Number of Monte-Carlo samples to determine
    :param statistic: Name of the statistic being tested (regression or correlation)

    Outputs:
    :param pval: Two-tailed p-values of the statistic, indicating significance
    '''

    # Remove NaNs
    # y_filled = np.where(np.isnan(y), 0, y)
    
    # Reshape to time x space if necessary
    # if len(x.shape > 1):
    #     T, I, J = x.shape
    #     x = x.reshape(T, I*J)

    # Determine the shape of the samples and outputs based on y
    if len(y.shape) > 1:
        T, IJ = y.shape
        # y = y.reshape(T, I*J)
        # slope = slope.reshape(IJ)
    else:
        IJ = 1
        T = x.size

    # Intialize the random samples of the statistic
    ind = np.random.randint(0, T, (N, T))
    # print(ind.shape)

    mc = np.ones((IJ, N)) * np.nan
    # print(mc.shape)

    # Make N random samples of the statistic with suffled values (based on ind)
    for n, i in tqdm(enumerate(ind)):
        if statistic == 'regression':
            # N samples of regression calculations
            mc[:,n], _, _ = least_squares(x, y[i,:]) if len(y.shape) > 1 else least_squares(x, y[i])
        elif statistic == 'correlation':
            # N samples of correlation calculations (note this process is very time consuming)
            mc[:,n] = stats.pearsonr(x, y[i,:], axis = 0).statistic if len(y.shape) > 1 else stats.pearsonr(x, y[i]).statistic
            # correlate(x, y[i,:]) if len(y.shape) > 1 else correlate(x, y[i])
    
    # Determine the p-value(s) based on where the original statistic is in the distribution of random samples
    if len(y.shape) > 1:
        pval = np.array([stats.percentileofscore(mc[i,:], original_statistic[i])/100 for i in range(original_statistic.size)])
    else:
        pval = np.array([stats.percentileofscore(mc[0,:], original_statistic)/100])

    return pval

    # # Calculate the correlation
    # if len(y.shape) > 1:
    #     R2 = np.nanvar(yhat - np.nanmean(y_filled, axis = 0), axis = 0)/(np.nanvar(y_filled, axis = 0)*np.nanvar(yhat, axis = 0)+1e-5)
    # else:
    #     R2 = np.nanvar(yhat - np.nanmean(y_filled))/(np.nanvar(y_filled)*np.nanvar(yhat)+1e-5)
    # corr = np.sign(slope) * np.sqrt(R2)

    # # Compute the effective sample size
    # # First get the lag-1 autocorrelations in x (r1) and y (r2)
    # T = x.shape[0]
    # xmean1 = np.nanmean(x[:-1]); xmean2 = np.nanmean(x[1:])
    # r1 = np.dot((x[:-1]-xmean1).T, (x[1:]-xmean2))/((T-1) * np.nanstd(x[:-1], axis = 0) * np.nanstd(x[1:], axis = 0))
    # if len(y.shape) > 1:
    #     r2 = np.diag(np.dot(y_filled[:-1,...].T, y_filled[1:,...]))/((T-1) * np.nanstd(y_filled[:-1,...], axis = 0) * np.nanstd(y_filled[1:,...], axis = 0)+1e-5)
    # else:
    #     r2 = np.nansum(y_filled[:-1] * y_filled[1:])/((T-1) * np.nanstd(y_filled[:-1]) * np.nanstd(y_filled[1:] + 1e-5))

    # # Next determine the effective sample size Neff
    # Neff = T * (1 - r1*r2)/(1 + r1*r2)
    # print(corr, r2)
    # # Calculate the t-statistic
    # t = corr * np.sqrt(Neff - 2)/(np.sqrt(1 - (corr**2)))

    # # Obtain the p-value
    # pval = 2*stats.t.cdf(-abs(t), Neff-2)
    # print(np.nanmean(pval), np.nanmean(t), np.nanmean(corr), np.nanmean(r2))
    # # Reshape pval onto a grid if necessary
    # # if len(y.shape) > 1:
    # #     pval = pval.shape(I, J)

    # return pval