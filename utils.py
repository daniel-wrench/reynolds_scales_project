import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cdflib
from pprint import pprint
#venv on Raapoi struggles to install this

import statsmodels.api as sm

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


def read_cdf(cdf_file_path: str) -> cdflib.cdfread.CDF:
    """
    Read a .cdf file as a cdf file object
    args:
      cdf_file_path: path to cdf
    """
    try:
        cdf_file_object = cdflib.CDF(cdf_file_path)
    except Exception:
        raise Exception('Exception while reading file')
    return cdf_file_object


def convert_cdf_to_dataframe(cdf_file_object: cdflib.cdfread.CDF, varlist=None) -> pd.DataFrame:
    """
    Convert a cdf file object to a Pandas DataFrame.
    args:
      varlist: list of strings. Specify the variables to include in the resulting DataFrame as they appear in the .cdf file.
               Multi-dimensional attributes are split into multiple columns with the names attribute_x
    """
    if varlist == None:
        varlist = cdf_file_object.cdf_info(
        )['rVariables'] + cdf_file_object.cdf_info()['zVariables']
    variables_to_read = varlist.copy()
    for var_name in varlist:
        if var_name not in cdf_file_object.cdf_info()['rVariables'] + cdf_file_object.cdf_info()['zVariables']:
            print(f'variable name "{var_name}" not in cdf file; skipping it')
            variables_to_read.remove(var_name)
    result_dict = {}
    for var_name in variables_to_read:
        variable_values = cdf_file_object.varget(var_name)
        if len(variable_values.shape) == 1:  # single attribute
            result_dict.update({var_name: variable_values})
        else:  # multi-dimensional attribute
            expanded_dict = {f'{var_name}_{dim_index}': variable_values[:, dim_index] for dim_index in range(
                variable_values.shape[1])}
            result_dict.update(expanded_dict)
    return pd.DataFrame(result_dict)


def replace_fill_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Replace dataframe fill values with numpy NaN. Fill values are documented in https://omniweb.gsfc.nasa.gov/html/omni_min_data.html.
    args:
      dataframe: pandas dataframe to remove fill values from
    """
    df_cleaned = dataframe.replace({9.9: np.nan, 999: np.nan, 999.99: np.nan, 999999: np.nan,
                       99.99: np.nan, 9999.99: np.nan, 9999999.: np.nan, 99999.99: np.nan,
                       99999.9: np.nan, 99999: np.nan}, inplace=False)
    return df_cleaned


def format_epochs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Epoch attribute into a DateTime object. Epoch column must exist.
    args:
      dataframe: pandas dataframe to format
    """
    assert type(
        dataframe) == pd.core.frame.DataFrame, 'Input is not of type DataFrame'
    assert 'Epoch' in dataframe.columns or 'EPOCH' in dataframe.columns, 'Epoch column does not exist'
    result_dataframe = dataframe.copy()
    result_dataframe['Epoch'] = result_dataframe['Epoch'].apply(
        lambda x: cdflib.epochs.CDFepoch.to_datetime(x, to_np=True)[0])
    result_dataframe.rename({'Epoch': 'Timestamp'},
                            axis='columns', inplace=True)
    return result_dataframe


def resample_time_series(dataframe: pd.DataFrame, cadence: str = '5T', agg_types: dict = None) -> pd.DataFrame:
    """
    Resample time series data to a regular cadence in order to merge data from different sources.
    args:
      dataframe: time series dataframe slice
      cadence: time cadence to resample on
      agg_types: aggregation types specified for all variables, recommended stay as mean
    """
    if not cadence:
        return dataframe.set_index('Timestamp')
    assert type(
        dataframe) == pd.core.frame.DataFrame, 'Input is not of type DataFrame'
    assert 'Timestamp' in dataframe.columns, 'Timestamp column does not exist'
    if not agg_types:
        return dataframe.resample(cadence, on='Timestamp').mean()
    return dataframe.resample(cadence, on='Timestamp').agg(agg_types)


def join_dataframes_on_timestamp(dataframe_1: pd.DataFrame, dataframe_2: pd.DataFrame) -> pd.DataFrame:
    """
    Join dataframes on the Timestamp attribute. Dataframes should be (re)sampled on the same cadence.
    args:
      dataframe_1 and dataframe_2: dataframes to join, must contain a synched, formatted datetime column
    """
    assert type(
        dataframe_1.index) == pd.core.indexes.datetimes.DatetimeIndex, 'dataframe_1 does not have a Datetime index'
    assert type(
        dataframe_2.index) == pd.core.indexes.datetimes.DatetimeIndex, 'dataframe_2 does not have a Datetime index'
    joined_df = pd.merge(dataframe_1, dataframe_2, how='left', on='Timestamp')
    return joined_df


def mask_and_interpolate_outliers(dataframe: pd.DataFrame, threshold_dict: dict) -> pd.DataFrame:
    """
    Mask values outside of the specified thresholds with numpy NaNs. Interpolate NaNs
    inputs:
      threshold_dict: dict. Pairs of column names and thresholds to identify outliers.
    """
    # if more than 40% of values in any column are missing, skip data period
    if (dataframe.isna().sum()/dataframe.shape[0] >= 0.4).any():
        return None
    if not threshold_dict:
        dataframe.iloc[:,1:] = dataframe.iloc[:,1:].interpolate()
        return dataframe
        # Selected only non-time columns due to issues with pd.DataFrame.interpolate not handling certain time columns
    result_dataframe = dataframe.copy()
    for column_name, thresholds in threshold_dict.items():
        result_dataframe[column_name] = result_dataframe[column_name].mask((result_dataframe[column_name] <= thresholds[0]) | (
            result_dataframe[column_name] >= thresholds[-1])).interpolate()
    return result_dataframe


def estimate_second_derivative(autocorrelation_function_array: np.ndarray, limit: int = 10, steps: int = 50, only_origin: bool = False) -> np.ndarray:
    """
    Estimate the second derivative of the autocorrelation function near the origin.
    inputs:
      limit: int. Defines the neighborhood around the origin to fit splines on. Recommended 10
      steps: int. Defines the resolution of the second derivative function.
             Mainly used for plotting. Recommended about 5X the limit.
      only_origin: bool. Ignore plotting and only output value at origin.
    """
    y_splines_fit = UnivariateSpline(range(len(autocorrelation_function_array)),
                                     autocorrelation_function_array,
                                     s=0,
                                     k=3)
    if only_origin:
        return y_splines_fit.derivative(n=2)(0)
    x_range = np.linspace(0, limit, steps)
    second_derivative = y_splines_fit.derivative(n=2)(x_range)
    return second_derivative


def pipeline(file_path, varlist, cadence='5T', thresholds={}):
    """
    pipeline of all functions to streamline run file
    """
    try:
        df = convert_cdf_to_dataframe(read_cdf(file_path), varlist=varlist)
        df = replace_fill_values(df)
        df = format_epochs(df)
        df = mask_and_interpolate_outliers(df, thresholds)
        if df is None or df.shape[0] == 0:
            print('dataframe had too many missing values')
            return None
        if cadence not in [0, '0S', None]:
            df = resample_time_series(
                df, cadence=cadence).interpolate().ffill().bfill()
        # df.set_index('Timestamp', inplace=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def print_cdf_info(file_path: str) -> None:
    """
    Print the cdf info of the specified file, for debugging purposes
    """
    pprint(read_cdf(file_path).cdf_info())


def compute_autocorrelation_function(dataframe: pd.DataFrame, number_of_lags: int) -> np.ndarray:
    """
    Compute autocorrelation function of time series data with specified number of lags.
    args:
      number_of_lags: number of lags to compute autocorrelation for; will differ for both length scales
    """
    # autocorrelation = sm.tsa.acf(dataframe[column_name] - np.mean(dataframe[column_name]), fft=False, nlags=number_of_lags)
    autocorrelation = sm.tsa.acf(dataframe['BGSE_0'], fft=False, nlags=number_of_lags) + sm.tsa.acf(
        dataframe['BGSE_1'], fft=False, nlags=number_of_lags) + sm.tsa.acf(dataframe['BGSE_2'], fft=False, nlags=number_of_lags)
    autocorrelation /= 3
    return autocorrelation


def exp_fit(r, lambda_c):
    """
    fit function for determining correlation scale, through the optimal lambda_c value
    """
    return np.exp(-1*r/lambda_c)


def para_fit(x, a):
    """
    fit function for determining taylor scale, through the optimal lambda_c value
    """
    return a*x**2 + 1

# OLD METHOD

# def compute_taylor_correlation_time_scales(dataframe, ind_1, ind_2, sample_interval, num_lags=100, show=False):
#   if show:
#     print(f'{ind_1} to {ind_2}')
#   ts = dataframe[ind_1:ind_2]
#   ac = compute_autocorrelation_function(ts, num_lags)
#   ac_x = np.array(range(len(ac)))

#   ac_x = ac_x*sample_interval
#   num_seconds_for_lambda_t_fit = 2
#   num_lags_for_lambda_t_fit = int(num_seconds_for_lambda_t_fit/sample_interval)
#   print(num_lags_for_lambda_t_fit)
#   num_seconds_for_lambda_c_fit = 20
#   num_lags_for_lambda_c_fit = int(num_seconds_for_lambda_c_fit/sample_interval)
#   c_opt, c_cov = curve_fit(exp_fit, ac_x[:num_lags_for_lambda_c_fit], ac[:num_lags_for_lambda_c_fit], 1000)
#   t_opt, t_cov = curve_fit(para_fit, ac_x[:num_lags_for_lambda_t_fit], ac[:num_lags_for_lambda_t_fit], 100)
#   # c_opt, c_cov = curve_fit(exp_fit, ac_x[:len(ts)//20], ac[:len(ts)//20], 100)
#   # t_opt, t_cov = curve_fit(para_fit, ac_x[:len(ts)//100], ac[:len(ts)//100], 100)
#   lambda_c = c_opt[0]
#   lambda_t = (-1*t_opt[0])**-.5
#   if show:
#     ax = plt.gca()
#     ax.set_ylim(-.2, 1.2)
#     plt.plot(ac_x, ac)
#     plt.plot(np.array(range(int(1.1*lambda_c))), exp_fit(np.array(range(int(1.1*lambda_c))), *c_opt), 'r-')
#     plt.plot(np.array(range(int(1.1*lambda_t))), para_fit(np.array(range(int(1.1*lambda_t))), *t_opt), 'g-')
#     box_color = 'grey' if lambda_c > lambda_t else 'red'
#     ax.text(ac_x[-1]*(12/10), .9, f'lambda_c: {round(lambda_c, 1)} [sec]\nlambda_t: {round(lambda_t, 1)} [sec]',style='italic', fontsize=10,
#           bbox={'facecolor': box_color, 'alpha': 0.5, 'pad': 10})
#     plt.show()

#     print(f'correlation (time) scale: {round(lambda_c, 3)} [sec]')
#     print(f'transverse taylor (time) scale: {round(lambda_t, 3)} [sec]')
#   return ind_1, round(lambda_c, 5), round(lambda_t, 5), sample_interval

# previous version called estimate_correlation_scale()
def compute_outer_scale_exp_trick(autocorrelation_x: np.ndarray, autocorrelation_y: np.ndarray, show = False):
    """
    computes the correlation scale through the "1/e" estimation method.
    autocorrelation_x assumed already in time scale
    """
    for i, j in zip(autocorrelation_y, autocorrelation_x):
        if i <= np.exp(-1):
            # print(i, j)
            idx_2 = np.where(autocorrelation_x == j)[0]
            idx_1 = idx_2 - 1
            x2 = autocorrelation_x[idx_2]
            x1 = autocorrelation_x[idx_1]
            y1 = autocorrelation_y[idx_1]
            y2 = autocorrelation_y[idx_2]
            x_opt = x1 + ((y1 - np.exp(-1))/(y1-y2))*(x2-x1)
            # print(autocorrelation_x[idx_1], autocorrelation_y[idx_1])
            # print(autocorrelation_x[idx_2], autocorrelation_y[idx_2])
            # print('e:', np.exp(-1))
            # print(x_opt)

            try:

                # Optional plotting
                if show == True:

                    dt = autocorrelation_x[1]-autocorrelation_x[0]

                    fig, ax = plt.subplots(constrained_layout=True)
                    ax.plot(autocorrelation_x, autocorrelation_y)
                    ax.set_xlabel('$\\tau$ (sec)')
                    ax.set_ylabel('Autocorrelation')

                    # For plotting secondary axes
                    def sec2lag(x):
                        return x / dt

                    def lag2sec(x):
                        return x * dt

                    secax_x = ax.secondary_xaxis('top', functions=(sec2lag, lag2sec))
                    secax_x.set_xlabel('$\\tau$ (lag)')

                    def sec2km(x):
                        return x * 400

                    def km2sec(x):
                        return x / 400

                    # use of a float for the position:
                    secax_x2 = ax.secondary_xaxis(-0.2, functions=(sec2km, km2sec))
                    secax_x2.set_xlabel('$r$ (km)')

                    plt.axhline(np.exp(-1), color = 'black')
                    plt.axvline(x_opt[0], color = 'black')
                    plt.show()

                return round(x_opt[0], 3)


            except Exception:
                return 0

    # none found
    return -1


def compute_taylor_time_scale(dataframe, ind_1, ind_2, sample_interval, num_lags=100, show=False):
    """
    computes the taylor scale through the parabolic fitting method.
    args:
      ind_1 and ind_2: indices to slice dataframe
      sample_interval: to convert from lag space to time scale
      show: only True for debugging, shows plots for verification purposes
    """
    try:
        if show:
            print(f'{ind_1} to {ind_2}')
        ts = dataframe[ind_1:ind_2]
        ac = compute_autocorrelation_function(ts, num_lags)
        ac_x = np.array(range(len(ac)))
        ac_x = ac_x*sample_interval
        num_seconds_for_lambda_t_fit = 2
        num_lags_for_lambda_t_fit = int(num_seconds_for_lambda_t_fit/sample_interval)
        # print(num_lags_for_lambda_t_fit)
        t_opt, t_cov = curve_fit(
            para_fit, ac_x[:num_lags_for_lambda_t_fit], ac[:num_lags_for_lambda_t_fit], 10)
        lambda_t = (-1*t_opt[0])**-.5
        print(ind_1, t_opt, lambda_t)
        if show:
            ax = plt.gca()
            ax.set_ylim(-.2, 1.2)
            plt.plot(ac_x, ac)
            plt.plot(
                np.array(range(int(1.1*lambda_t))),
                para_fit(np.array(range(int(1.1*lambda_t))), *t_opt),
                'g-')
            box_color = 'grey' if lambda_t < 50 else 'red'
            ax.text(
                ac_x[-1]*(8/10), 
                .9, 
                f'lambda_t: {round(lambda_t, 1)} [sec]', 
                style='italic', 
                fontsize=10,
                bbox={'facecolor': box_color, 'alpha': 0.5, 'pad': 10})
            plt.show()

            print(
                f'transverse taylor (time) scale: {round(lambda_t, 3)} [sec]')
            print('')
        return ind_1, round(lambda_t, 5), sample_interval
    except Exception:
        return ind_1, -1, sample_interval


def compute_correlation_time_scale(dataframe, ind_1, ind_2, sample_interval, num_lags=100, show=False):
    """
    computes the correlation scale through the two methods: 1/e estimation and exponential fitting.
    args:
      ind_1 and ind_2: indices to slice dataframe
      sample_interval: to convert from lag space to time scale
      show: only True for debugging, shows plots for verification purposes
    """
    try:
        if show:
            print(f'{ind_1} to {ind_2}')
        ts = dataframe[ind_1:ind_2]
        ac = compute_autocorrelation_function(ts, num_lags)
        ac_x = np.array(range(len(ac)))

        ac_x = ac_x*sample_interval
        num_seconds_for_lambda_c_fit = 1000

        # estimate using 1/e trick
        lambda_c_estimate = estimate_correlation_scale(ac_x, ac)

        num_lags_for_lambda_c_fit = int(
            num_seconds_for_lambda_c_fit/sample_interval)
        # print(num_lags_for_lambda_c_fit)
        c_opt, c_cov = curve_fit(
            exp_fit, ac_x[:num_lags_for_lambda_c_fit], ac[:num_lags_for_lambda_c_fit], 1000)
        lambda_c = c_opt[0]
        if show:
            ax = plt.gca()
            ax.set_ylim(-.2, 1.2)
            plt.plot(ac_x, ac)
            plt.plot(
                np.array(range(int(num_seconds_for_lambda_c_fit))), 
                exp_fit(
                    np.array(range(int(num_seconds_for_lambda_c_fit))),
                     *c_opt
                     ), 
                     'r-')
            box_color = 'grey' if lambda_c > 50 else 'red'
            ax.text(ac_x[-1]*(5/10), 0.9, f'lambda_c: {round(lambda_c, 1)} [sec]\nestimate: {round(lambda_c_estimate, 1)} [sec]', style='italic', fontsize=10,
                    bbox={'facecolor': box_color, 'alpha': 0.5, 'pad': 10})
            plt.show()

            print(f'correlation (time) scale: {round(lambda_c, 3)} [sec]')
        return ind_1, round(lambda_c, 5), round(lambda_c_estimate, 5), sample_interval
    except Exception:
        return ind_1, -1, -1, sample_interval
