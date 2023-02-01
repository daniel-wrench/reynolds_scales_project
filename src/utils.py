import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cdflib
import statsmodels.api as sm
from pprint import pprint
from scipy.optimize import curve_fit

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


def resample_time_series(dataframe: pd.DataFrame, cadence: str, agg_types: dict = None) -> pd.DataFrame:
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
    joined_df = pd.merge(dataframe_1, dataframe_2, how='outer', on='Timestamp')
    return joined_df


def mask_outliers(dataframe: pd.DataFrame, threshold_dict: dict) -> pd.DataFrame:
    """
    Mask values outside of the specified thresholds with numpy NaNs. Interpolate NaNs
    inputs:
      threshold_dict: dict. Pairs of column names and thresholds to identify outliers.
    """
    if not threshold_dict:
        return dataframe
        # Selected only non-time columns due to issues with pd.DataFrame.interpolate not handling certain time columns
    result_dataframe = dataframe.copy()
    for column_name, thresholds in threshold_dict.items():
        result_dataframe[column_name] = result_dataframe[column_name].mask((result_dataframe[column_name] <= thresholds[0]) | (
            result_dataframe[column_name] >= thresholds[-1]))
    return result_dataframe

def pipeline(file_path, varlist, cadence=None, thresholds={}):
    """
    pipeline of all functions to streamline run file
    """
    try:
        df = convert_cdf_to_dataframe(read_cdf(file_path), varlist=varlist)
        df = replace_fill_values(df)
        df = format_epochs(df)
        df = mask_outliers(df, thresholds)
        if cadence not in [0, '0S', None]:
            df = resample_time_series(df, cadence=cadence)
        # df.set_index('Timestamp', inplace=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

def print_cdf_info(file_path: str) -> None:
    """
    Print the cdf info of the specified file, for debugging purposes
    """
    pprint(read_cdf(file_path).cdf_info())

# Calculate 3D power spectrum
import scipy.signal as signal

def SmoothySpec(a,nums=None):
   """Smooth a curve using a moving average smoothing"""
   b=a.copy()
   if nums is None: nums=2*len(b)//3
   for i in range(nums):
      b[i+1:-1] = 0.25*b[i:-2]+0.5*b[i+1:-1]+0.25*b[i+2:]
   return b

def fitpowerlaw(ax,ay,xi,xf):
   idxi=np.argmin(abs(ax-xi))
   idxf=np.argmin(abs(ax-xf))
   xx=np.linspace(xi,xf,100)
   z=np.polyfit(np.log(ax[idxi:idxf]),np.log(ay[idxi:idxf]),1);
   p=np.poly1d(z);
   pwrl=np.exp(p(np.log(xx)))
   return z,xx,pwrl

def compute_spectral_stats(np_array, dt, f_min_inertial, f_max_inertial, f_min_kinetic, f_max_kinetic, di = None, velocity = None, plot=False):
    """ Compute the autocorrelation function for a scalar or vector time series.
    
    ### Args:

    - np_array: Array of shape (1,n) or (3,n)
    - dt: Cadence of measurements, or time between each sample: one sample every dt seconds
    - di: (Optional, only used for plotting) Ion inertial length in km
    - velocity: (Optional, only used for plotting) Solar wind velocity in km/s
    - corr_scale: (Optional, only used for plotting) Correlation scale (seconds)
    - taylor_scale: (Optional, only used for plotting) Taylor scale (seconds)
    ### Returns:

    - z_i: Slope in the inertial range
    - z_k: Slope in the kinetic range
    - spectral_break: Frequency of the spectral break between the two ranges

    """
    x_freq = 1/dt

    f_periodogram, power_periodogram_0 = signal.periodogram(np_array[0], fs = x_freq, window="boxcar", scaling="density")
    power_periodogram_0 = (x_freq/2)*power_periodogram_0

    f_periodogram, power_periodogram_1 = signal.periodogram(np_array[1], fs = x_freq, window="boxcar", scaling="density")
    power_periodogram_1 = (x_freq/2)*power_periodogram_1

    f_periodogram, power_periodogram_2 = signal.periodogram(np_array[2], fs = x_freq, window="boxcar", scaling="density")
    power_periodogram_2 = (x_freq/2)*power_periodogram_2

    power_periodogram = (power_periodogram_0 + power_periodogram_1 + power_periodogram_2)/3

    p_smooth = SmoothySpec(power_periodogram) # Slowest part of this function - takes ~ 10 seconds

    zk, xk, pk = fitpowerlaw(f_periodogram, p_smooth, f_min_kinetic, f_max_kinetic) # Kinetic range
    zi, xi, pi = fitpowerlaw(f_periodogram, p_smooth, f_min_inertial, f_max_inertial) # Inertial range

    try:
        powerlaw_intersection = np.roots(zk-zi)
        spectral_break = np.exp(powerlaw_intersection)
    except:
        print("could not compute power-law intersection")
        spectral_break = [np.nan]

    if round(spectral_break[0], 4) == 0 or spectral_break[0] > 1:
       spectral_break = [np.nan]
        
    if plot == True:
        fig, ax = plt.subplots(figsize = (7,4))
        ax.semilogy(f_periodogram, power_periodogram, label = "Raw periodogram")
        ax.semilogy(f_periodogram, p_smooth, label = "Smoothed periodogram", color = "cyan")
        ax.semilogy(xi, pi, c = "red", label = "Inertial range power-law fit: $\\alpha_i$ = {0:.2f}".format(zi[0]))
        ax.semilogy(xk, pk, c = "yellow", label = "Kinetic range power-law fit: $\\alpha_k$ = {0:.2f}".format(zk[0]))
        ax.semilogx()
        if spectral_break[0] is not np.nan:
            ax.axvline(np.exp(np.roots(zk-zi)), color = "black", label = "Spectral break: $f_d={0:.2f}$".format(spectral_break[0]))

        # Adding in proton inertial frequency
        if di is not None and velocity is not None:
            f_di = velocity/(2*np.pi*di)
            ax.axvline(f_di, color = "green", label = "Proton inertial frequency: $f_{di}=$" + "{0:.2f}".format(f_di))

        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('PSD')
        ax.legend()
        #plt.grid()
        #plt.show()

        return zi[0], zk[0], spectral_break[0], fig, ax
    else:
        return zi[0], zk[0], spectral_break[0]

def compute_nd_acf(np_array, nlags, dt, show=False):
    """ Compute the autocorrelation function for a scalar or vector time series.

    Args:

    - np_array: Array of shape (1,n) or (3,n)
    - nlags: The number of lags to calculate the ACF up to
    - dt: Cadence of measurements, or time between each sample: one sample every dt seconds

    Returns:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - R: The values of the ACF from lag 0 to nlags

    """

    # Previously Kevin had fft=False - this was far slower
    if np_array.shape[0] == 3:
        acf = \
            sm.tsa.acf(np_array[0], fft=True, nlags=nlags) + \
            sm.tsa.acf(np_array[1], fft=True, nlags=nlags) + \
            sm.tsa.acf(np_array[2], fft=True, nlags=nlags)
        acf /= 3

    elif np_array.shape[0] == 1:
        acf = sm.tsa.acf(np_array[0], fft=True, nlags=nlags)

    else:
        raise ValueError(
            "Array is not 3D or 1D. If after a 1D acf, try putting square brackets around the pandas series in np.array()")

    time_lags = np.arange(0, nlags+1)*dt

    # Optional plotting
    if show == True:

        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(time_lags, acf)
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

        plt.show()

    return time_lags, acf

# previous version called estimate_correlation_scale()
def compute_outer_scale_exp_trick(autocorrelation_x: np.ndarray, autocorrelation_y: np.ndarray, plot=False):
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

                # Optional plotting, set up to eventually display all 3 corr scale methods
                if plot == True:

                    dt = autocorrelation_x[1]-autocorrelation_x[0]

                    fig, ax = plt.subplots(1, 3, figsize = (9, 4), constrained_layout=True)
                    
                    ax[0].plot(autocorrelation_x, autocorrelation_y)
                    ax[0].set_xlabel('$\\tau$ (sec)')
                    ax[0].set_ylabel('Autocorrelation')

                    # For plotting secondary axes
                    def sec2lag(x):
                        return x / dt

                    def lag2sec(x):
                        return x * dt

                    secax_x = ax[0].secondary_xaxis('top', functions=(sec2lag, lag2sec))
                    secax_x.set_xlabel('$\\tau$ (lag)')

                    def sec2km(x):
                        return x * 400

                    def km2sec(x):
                        return x / 400

                    # use of a float for the position:
                    secax_x2 = ax[0].secondary_xaxis(-0.2, functions=(sec2km, km2sec))
                    secax_x2.set_xlabel('$r$ (km)')

                    ax[0].axhline(np.exp(-1), color = 'black')
                    ax[0].axvline(x_opt[0], color = 'black')

                    return round(x_opt[0], 3), fig, ax
                else:
                    return round(x_opt[0], 3)
            except Exception:
                return 0

    # none found
    return -1

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


def compute_outer_scale_exp_fit(time_lags, acf, seconds_to_fit, fig=None, ax=None, plot=False):

    dt = time_lags[1]-time_lags[0]
    num_lags_for_lambda_c_fit = int(seconds_to_fit/dt)
    c_opt, c_cov = curve_fit(
        exp_fit, time_lags[:num_lags_for_lambda_c_fit], acf[:num_lags_for_lambda_c_fit], p0=1000)
    lambda_c = c_opt[0]

    
    # Optional plotting 
    if plot == True:
        if fig is not None and ax is not None:
            fig = fig
            ax = ax
            column = 1

        ax[column].plot(time_lags, acf, label = "Autocorrelation")
        ax[column].plot(
            np.array(range(int(seconds_to_fit))),
            exp_fit(
                np.array(range(int(seconds_to_fit))),
                *c_opt
            ),
            label = "Exponential fit")
        ax[column].set_xlabel('$\\tau$ (sec)')
        ax[column].set_ylabel('Autocorrelation')

        # For plotting secondary axes
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x = ax[column].secondary_xaxis('top', functions=(sec2lag, lag2sec))
        secax_x.set_xlabel('$\\tau$ (lag)')

        def sec2km(x):
            return x * 400

        def km2sec(x):
            return x / 400

        # use of a float for the position:
        secax_x2 = ax[column].secondary_xaxis(-0.2, functions=(sec2km, km2sec))
        secax_x2.set_xlabel('$r$ (km)')

        #ax[1].legend(loc='center right')
        #ax[1].set_title("{}: {:.2f}".format(figname, lambda_c))

        return lambda_c, fig, ax
    else:
        return lambda_c

def compute_outer_scale_integral(time_lags, acf, fig=None, ax=None, plot=False):

    dt = time_lags[1]-time_lags[0]
    idx = np.argmin(np.abs(acf)) # Getting the index where the ACF falls to 0
    int = np.sum(acf[:idx])*dt # Computing integral up to that index

    # Optional plotting
    if plot == True:
        # Optional plotting 
        if fig is not None and ax is not None:
            fig = fig
            ax = ax
            column = 2
        #ax.set_ylim(-.2, 1.2)
        ax[column].plot(time_lags, acf, label="Autocorrelation")
        ax[column].fill_between(time_lags, 0, acf, where=acf > 0)
        # box_color = 'grey' if lambda_c > 50 else 'red'
        # ax.text(time_lags[-1]*(5/10), 0.9, f'$\lambda_c$: {round(lambda_c, 1)}s', style='italic', fontsize=10,
        #         bbox={'facecolor': box_color, 'alpha': 0.5, 'pad': 10})
        ax[column].set_xlabel('$\\tau$ (sec)')
        ax[column].set_ylabel('Autocorrelation')

        # For plotting secondary axes
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x = ax[column].secondary_xaxis('top', functions=(sec2lag, lag2sec))
        secax_x.set_xlabel('$\\tau$ (lag)')

        def sec2km(x):
            return x * 400

        def km2sec(x):
            return x / 400

        # use of a float for the position:
        secax_x2 = ax[column].secondary_xaxis(-0.2, functions=(sec2km, km2sec))
        secax_x2.set_xlabel('$r$ (km)')

        return int, fig, ax
    else:
        return int
        

def compute_taylor_scale(time_lags, acf, tau_fit, plot=False, show_intercept = False):
    """Compute the Taylor microscale

    Args:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - acf: The y-values of the ACF
    - tau_fit: number of lags to fit the parabola over
    """

    # If using seconds_fit as the fitting argument instead:

    dt = time_lags[1]-time_lags[0]
    # tau_fit = int(seconds_fit/dt)

    t_opt, t_cov = curve_fit(
        para_fit,
        time_lags[:tau_fit],
        acf[:tau_fit],
        p0=10) # Initial guess for the parameters
    lambda_t = (-1*t_opt[0])**-.5

    extended_parabola_x = np.arange(0, 1.2*lambda_t, 0.1)
    extended_parabola_y = para_fit(extended_parabola_x, *t_opt)

    # Optional plotting set up to eventually show Chuychai correction factor in second panel
    if plot == True:

        #mpl_fig = plt.figure()

        fig, ax = plt.subplots(1,2, figsize = (9,4), constrained_layout=True)

        ax[0].scatter(time_lags, acf, label="Autocorrelation", s = 0.5)
        ax[0].plot(
            extended_parabola_x,
            extended_parabola_y,
            '-y',
            label="Parabolic fit")
        #plt.axhline(0, color = 'black')
        ax[0].axvline(tau_fit*(time_lags[1]-time_lags[0]), color='purple', label = "$\\tau_{fit}=20$ lags")

        ax[0].set_xlim(-0.2, tau_fit*dt*2)
        ax[0].set_ylim(0.99, 1.001)

        if show_intercept == True:
            ax[0].set_ylim(0, 1.05)
            ax[0].set_xlim(-2, lambda_t + 5)

        ax[0].set_xlabel('$\\tau$ (sec)')
        ax[0].set_ylabel('Autocorrelation')

        # For plotting secondary axes
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x = ax[0].secondary_xaxis('top', functions=(sec2lag, lag2sec))
        secax_x.set_xlabel('$\\tau$ (lag)')

        def sec2km(x):
            return x * 400

        def km2sec(x):
            return x / 400

        # use of a float for the position:
        secax_x2 = ax[0].secondary_xaxis(-0.2, functions=(sec2km, km2sec))
        secax_x2.set_xlabel('$r$ (km)')
        
        ax[0].legend(loc = "center right")

        return lambda_t, fig, ax

    else:
        return lambda_t

def compute_taylor_chuychai(time_lags, acf, tau_min, tau_max, fig=None, ax=None, q=None, save=False, figname=""):
    """Compute a refined estimate of the Taylor microscale using a linear extrapolation method from Chuychai et al. (2014).

    Args:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - acf: The y-values of the ACF
    - tau_min: Minimum value for the upper lag to fit the parabola over. This should not be too small, because the data has finite time resolution and there may be limited data available at the shortest time lags. (You will see divergent behaviour if this happens.)
    - tau_max: Maximum value for the upper lag to fit the parabola over
    - q: Slope of the dissipation range
    """

    dt = time_lags[1]-time_lags[0]

    tau_fit = np.arange(tau_min, tau_max+1)
    tau_ts = np.array([])

    for i in tau_fit:
        lambda_t = compute_taylor_scale(time_lags, acf, tau_fit=i)
        tau_ts = np.append(tau_ts, lambda_t)

    # Performing linear extrapolation back to tau_fit = 0
    z, cov = np.polyfit(x=tau_fit, y=tau_ts, deg=1, cov=True)
    f = np.poly1d(z)

    ts_est_extra = z[1] # Extracting y-intercept
    
    # Getting standard deviation of y-intercept
    # (will plot +- 1 standard deviation)
    ts_est_extra_std = np.sqrt(cov[1,1])

    # Getting extrapolation line for plotting
    other_x = np.arange(0, tau_max+1)
    other_y = f(other_x)

    # Applying correction factor q from Chuychai et al. (2014)
    if q is not None:
        q_abs = np.abs(q)
        if q_abs < 2:
            r = -0.64*(1/q_abs)+0.72
        elif q_abs >= 2 and q_abs < 4.5:
            r = -2.61*(1/q_abs)+1.7
        elif q_abs >= 4.5:
            r = -0.16*(1/q_abs)+1.16

    else:
        r = 1

    ts_est = r*ts_est_extra 
    ts_est_std = r*ts_est_extra_std

    # Optional plotting
    if fig is not None and ax is not None:
        ax[1].plot(tau_fit*dt, tau_ts, color="blue",
                    label="Range of $\\tau_{TS}$ calculation")
        ax[1].plot(other_x*dt, other_y, color="black",
                    label="Linear extrapolation", ls='--')
        if q is not None:
            ax[1].plot(0, ts_est, "go", label = "Final estimate $\\tau_{{TS}}$ (q={0:.2f})".format(q))
            #ax[1].plot(0, ts_est_final_lower, "r+", markersize=1)
            #ax[1].plot(0, ts_est_final_upper, "r+", markersize=1)
        ax[1].set_xlabel("$\\tau_{fit}$ (sec)")
        ax[1].set_ylabel("$\\tau_{fit}^{est}$ (sec)")

        # For plotting secondary axes
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x = ax[1].secondary_xaxis('top', functions=(sec2lag, lag2sec))
        secax_x.set_xlabel('$\\tau_{fit}$ (lags)')

        def sec2km(x):
            return x * 400

        def km2sec(x):
            return x / 400

        # use of a float for the position:
        secax_x2 = ax[1].secondary_xaxis(-0.2, functions=(sec2km, km2sec))
        secax_x2.set_xlabel('$r_{fit}$ (km)')
        
        ax[1].legend()
        #plt.close()
        #ax.set_title("{}: {:.2f}".format(figname, ts_est))

        return ts_est, ts_est_std, fig, ax

    else:
        return ts_est, ts_est_std



