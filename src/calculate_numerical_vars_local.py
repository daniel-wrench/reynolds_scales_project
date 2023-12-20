
"""
calculate_numerical_vars.py

This script processes high-resolution and low-resolution wind magnetic field data to compute various metrics 
related to magnetic field fluctuations and their statistical properties.

Modules:
    - params: Contains parameters and configurations for data processing.
    - utils: Contains utility functions for data processing and analysis.

Steps:
    1. Read high-resolution and low-resolution wind magnetic field data from pickle files.
    2. Process data and initialize lists for storing computed metrics.
    3. Loop over each interval to a variety of quantities, some using mfi data, some using proton data, and some both. 
    This requires if-else logic to deal with missing data for each dataset in each interval. Basically, we linearly
    interpolate gaps if there is less than 10% missing, otherwise the corresponding computed statistics for that interval 
    are set to missing.
    4. Create a DataFrame containing the computed statistics for each interval.
    5. Save the final processed dataframes as pickle files to the directory data/processed/.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Custom modules
## Add src. prefix if running interactively
import params
import utils

print("\nREADING PICKLE FILES")

# High-res magnetic field data

df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + ".pkl")

df_wind_hr = df_wind_hr.rename(
    columns={
        params.Bx: "Bx",
        params.By: "By",
        params.Bz: "Bz"})

print("\nHigh-res Wind dataframe:\n")
print(df_wind_hr.info())
print(df_wind_hr.describe().round(2))

df_protons = pd.read_pickle("data/processed/" + params.proton_path + params.dt_protons + ".pkl")

df_protons = df_protons.rename(
    columns={
        params.Vx: "Vx",
        params.Vy: "Vy",
        params.Vz: "Vz",
        params.np: "np",
        params.nalpha: "nalpha",
        params.Tp: "Tp",
        params.Talpha: "Talpha"})

print("\n3s proton dataframe:\n")
print(df_protons.info())
print(df_protons.describe().round(2))

# Low-res magnetic field data

df_wind_lr = df_wind_hr.resample(params.dt_lr).mean()

# df_wind_lr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_lr + ".pkl")

# df_wind_lr = df_wind_lr.rename(
#     columns={
#         params.Bx: "Bx",
#         params.By: "By",
#         params.Bz: "Bz"})

print("\nLow-res Wind dataframe:\n")
print(df_wind_lr.info())
print(df_wind_lr.describe().round(2))

print("\nFINISHED READING PICKLE FILES")

# Initializing lists for storing computed metrics
timestamps = []

B0_list = []
db_list = []
db_a_list = []
dboB0_list = []
V0_list = []
v_r_list = []
dv_list = []
zp_list = []
zm_list = []
sigma_c_list = []
sigma_c_abs_list = []
sigma_r_list = []
ra_list = []
cos_a_list = []
np_list = []
nalpha_list = []
Tp_list = []
Talpha_list = []

inertial_slope_list = []
kinetic_slope_list = []
spectral_break_list = []
corr_scale_exp_fit_list = []
corr_scale_exp_trick_list = []
velocity_corr_scale_exp_trick_list = []
corr_scale_int_list = []
taylor_scale_u_list = []
taylor_scale_u_std_list = []
taylor_scale_c_list = []
taylor_scale_c_std_list = []

wind_df_hr_list_missing = []
wind_df_protons_list_missing = []

# For plotting ACFs (so only run on small number of intervals)
acf_hr_list = []
velocity_acf_lr_list = []
acf_lr_list = []

# Splitting entire dataframe into a list of intervals (each themselves a dataframe)
# First, set a start and end time for the first interval

starttime = df_wind_hr.index[0].round(params.int_size) 
# Will account for when the dataset does not start at a nice round 12H timestamp
# E.g. 2016-01-01 00:00

endtime = starttime + pd.to_timedelta(params.int_size) - pd.to_timedelta("0.01S")
# E.g. 2016-01-01 11:59:59.99

# Number of intervals in the dataset
n_int = np.round((df_wind_hr.index[-1]-df_wind_hr.index[0]) / pd.to_timedelta(params.int_size)).astype(int)

# NB: If we subset timestamps that don't exist in the dataframe, they will still be included in the list, just as
# missing dataframes. We can identify these with df.empty = True (or missing)

df = pd.DataFrame({
    "Timestamp": [np.nan]*n_int,
    "missing_mfi": [np.nan]*n_int,
    "missing_3dp": [np.nan]*n_int,
    "np": [np.nan]*n_int,
    "nalpha": [np.nan]*n_int,
    "Tp": [np.nan]*n_int,
    "Talpha": [np.nan]*n_int,
    "B0": [np.nan]*n_int,
    "db": [np.nan]*n_int,
    "db_a": [np.nan]*n_int,
    "dboB0": [np.nan]*n_int,
    "V0": [np.nan]*n_int,
    "v_r": [np.nan]*n_int,
    "dv": [np.nan]*n_int,
    "zp": [np.nan]*n_int,
    "zm": [np.nan]*n_int,
    "sigma_c": [np.nan]*n_int,
    "sigma_c_abs": [np.nan]*n_int,
    "sigma_r": [np.nan]*n_int,
    "ra": [np.nan]*n_int,
    "cos_a": [np.nan]*n_int,
    "qi": [np.nan]*n_int,
    "qk": [np.nan]*n_int,
    "fb": [np.nan]*n_int,
    "tcf": [np.nan]*n_int,
    "tce": [np.nan]*n_int,
    "tce_velocity": [np.nan]*n_int,
    "tci": [np.nan]*n_int,
    "ttu": [np.nan]*n_int,
    "ttu_std": [np.nan]*n_int,
    "ttc": [np.nan]*n_int,
    "ttc_std": [np.nan]*n_int
})

print("\nLOOPING OVER EACH INTERVAL")

for i in np.arange(n_int).tolist():
    int_start = starttime + i*pd.to_timedelta(params.int_size)
    int_end = (endtime + i*pd.to_timedelta(params.int_size))

    df.at[i, "Timestamp"] = int_start

    int_lr = df_wind_lr[int_start:int_end]
    int_hr = df_wind_hr[int_start:int_end]
    int_protons = df_protons[int_start:int_end]
    int_protons_lr = int_protons.resample(params.dt_lr).mean()

    # Record amount of missing data in each dataset in each interval
    if int_hr.empty:
        missing_mfi = 1
    else:
        missing_mfi = int_hr.iloc[:, 0].isna().sum()/len(int_hr)

    if int_protons.empty:
        missing_3dp = 1
    else:
        missing_3dp = int_protons.iloc[:, 0].isna().sum()/len(int_protons)

    # Save these values to their respective lists, to become columns in the final dataframe
    df.at[i, "missing_mfi"] = missing_mfi
    df.at[i, "missing_3dp"] = missing_3dp

    # What follows are nested if-else statements dealing with each combination of missing data between
    # the magnetic field and proton datasets, interpolating and calculating variables where possible
    # and setting them to missing where not 

    try: # try statement for error handling; see except statement at end of loop
        if missing_3dp <= 0.1:
            # Interpolate missing data, then fill any remaining gaps at start or end with nearest value
            int_protons = int_protons.interpolate(method="linear").ffill().bfill()
            int_protons_lr = int_protons_lr.interpolate(method="linear").ffill().bfill()

            df.at[i, "np"] = int_protons["np"].mean()
            df.at[i, "nalpha"] = int_protons["nalpha"].mean()
            df.at[i, "Tp"] = int_protons["Tp"].mean()
            df.at[i, "Talpha"] = int_protons["Talpha"].mean()

            Vx = int_protons["Vx"]
            Vy = int_protons["Vy"]
            Vz = int_protons["Vz"]

            Vx_mean = Vx.mean()
            Vy_mean = Vy.mean()
            Vz_mean = Vz.mean()

            # Save mean radial velocity (should dominate velocity mag)
            df.at[i, "v_r"] = np.abs(Vx_mean) # abs() because all vals negative (away from Sun)

            # Calculate velocity magnitude V0
            V0 = np.sqrt(Vx_mean**2+Vy_mean**2+Vz_mean**2)
            df.at[i, "V0"] = V0

            # Calculate rms velocity fluctuations, dv
            dvx = Vx - Vx_mean
            dvy = Vy - Vy_mean
            dvz = Vz - Vz_mean
            dv = np.sqrt(np.mean(dvx**2+dvy**2+dvz**2))
            df.at[i, "dv"] = dv

            # Compute proton velocity correlation length
            velocity_time_lags_lr, velocity_acf_lr = utils.compute_nd_acf(
                np.array([
                    int_protons_lr.Vx,
                    int_protons_lr.Vy,
                    int_protons_lr.Vz
                ]),
                nlags=params.nlags_lr,
                dt=float(params.dt_lr[:-1]))  # Removing "S" from end of dt string

            # velocity_acf_lr_list.append(velocity_acf_lr) #LOCAL ONLY
            velocity_corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(velocity_time_lags_lr, velocity_acf_lr)
            df.at[i, "tce_velocity"] = velocity_corr_scale_exp_trick

        if missing_mfi <= 0.1:

        # Interpolate missing data, then fill any remaining gaps at start or end with nearest value

            int_hr = int_hr.interpolate(method="linear").ffill().bfill() 
            int_lr = int_lr.interpolate(method="linear").ffill().bfill()

            # Firstly, do the calculations that we don't need proton data for
            # Then, condition on sufficient proton data and do the remainder

            # Resampling mag field data to 3s to match velocity data cadence
            ## NB: there is also a 3s cadence version of the mfi data, e.g. as used by Podesta2010, 
            ## but we want the highest res possible for the Taylor scale calculations

            Bx = int_hr["Bx"].resample(params.dt_protons).mean()
            By = int_hr["By"].resample(params.dt_protons).mean()
            Bz = int_hr["Bz"].resample(params.dt_protons).mean()

            Bx_mean = Bx.mean()
            By_mean = By.mean()
            Bz_mean = Bz.mean()

            # Calculate magnetic field magnitude B0
            B0 = np.sqrt(Bx_mean**2+By_mean**2+Bz_mean**2)
            df.at[i, "B0"] = B0

            # Calculate rms magnetic field fluctuations, db
            dbx = Bx - Bx_mean
            dby = By - By_mean
            dbz = Bz - Bz_mean
            db = np.sqrt(np.mean(dbx**2+dby**2+dbz**2))
            df.at[i, "db"] = db

            df.at[i, "dboB0"] = db/B0

            # Compute autocorrelations and power spectra
            time_lags_lr, acf_lr = utils.compute_nd_acf(
                np.array([
                    int_lr.Bx,
                    int_lr.By,
                    int_lr.Bz
                ]),
                nlags=params.nlags_lr,
                dt=float(params.dt_lr[:-1]))  # Removing "S" from end of dt string

            corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(time_lags_lr, acf_lr)
            df.at[i, "tce"] = corr_scale_exp_trick

            # Use estimate from 1/e method to select fit amount
            corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(
                time_lags_lr, acf_lr, np.round(2*corr_scale_exp_trick))
            df.at[i, "tcf"] = corr_scale_exp_fit

            corr_scale_int = utils.compute_outer_scale_integral(time_lags_lr, acf_lr)
            df.at[i, "tci"] = corr_scale_int

            time_lags_hr, acf_hr = utils.compute_nd_acf(
                np.array([
                    int_hr.Bx,
                    int_hr.By,
                    int_hr.Bz
                ]),
                nlags=params.nlags_hr,
                dt=float(params.dt_hr[:-1]))

            # ~1min per interval due to spectrum smoothing algorithm
            slope_i, slope_k, break_s = utils.compute_spectral_stats(
                np.array([
                    int_hr.Bx,
                    int_hr.By,
                    int_hr.Bz
                ]),
                dt=float(params.dt_hr[:-1]),
                f_min_inertial=params.f_min_inertial, f_max_inertial=params.f_max_inertial,
                f_min_kinetic=params.f_min_kinetic, f_max_kinetic=params.f_max_kinetic)

            df.at[i, "qi"] = slope_i
            df.at[i, "qk"] = slope_k
            df.at[i, "fb"] = break_s

            taylor_scale_u, taylor_scale_u_std = utils.compute_taylor_chuychai(
                time_lags_hr,
                acf_hr,
                tau_min=params.tau_min,
                tau_max=params.tau_max)

            df.at[i, "ttu"] = taylor_scale_u
            df.at[i, "ttu_std"] = taylor_scale_u_std


            taylor_scale_c, taylor_scale_c_std = utils.compute_taylor_chuychai(
                time_lags_hr,
                acf_hr,
                tau_min=params.tau_min,
                tau_max=params.tau_max,
                q=slope_k)

            df.at[i, "ttc"] = taylor_scale_c
            df.at[i, "ttc_std"] = taylor_scale_c_std

        if missing_3dp <= 0.1 and missing_mfi <= 0.1:
            # Already interpolated data, shouldn't need to do again

            ## Convert magnetic field fluctuations to Alfvenic units
            alfven_prefactor = 21.8/np.sqrt(int_protons["np"]) # Converting nT to Gauss and cm/s to km/s
            # note that Wang2012ApJ uses the mean density of the interval 

            dbx_a = dbx*alfven_prefactor
            dby_a = dby*alfven_prefactor
            dbz_a = dbz*alfven_prefactor
            db_a = np.sqrt(dbx_a**2+dby_a**2+dbz_a**2)
            db_a_rms = np.sqrt(np.mean(dbx_a**2+dby_a**2+dbz_a**2))
            df.at[i, "db_a"] = db_a_rms

            # Cross-helicity 
            Hc = np.mean(dvx*dbx_a + dvy*dby_a + dvz*dbz_a)

            # Normalize by energy (should then range between -1 and 1, like a normal correlation coefficient)
            # Only minor different calculating them separately, rather than np.mean(dv**2 + db**2)
            e_kinetic = np.mean(dv**2)
            e_magnetic = np.mean(db_a**2)

            sigma_c = 2*Hc/(e_kinetic+e_magnetic)
            df.at[i, "sigma_c"] = sigma_c
            df.at[i, "sigma_c_abs"] = np.abs(sigma_c)
            # Normalized residual energy
            sigma_r = (e_kinetic-e_magnetic)/(e_kinetic+e_magnetic)
            df.at[i, "sigma_r"] = sigma_r

            # Alfven ratio (ratio between kinetic and magnetic energy, typically ~0.5)
            ra = e_kinetic/e_magnetic
            df.at[i, "ra"] = ra

            # Alignment cosine (see Parashar2018PRL, ranges between -1 and 1)
            cos_a = Hc/np.mean(np.sqrt(e_kinetic*e_magnetic))
            df.at[i, "cos_a"] = cos_a

            # Elsasser variables
            zpx = dvx + dbx_a
            zpy = dvy + dby_a
            zpz = dvz + dbz_a
            zp = np.sqrt(np.mean(zpx**2+zpy**2+zpz**2))
            df.at[i, "zp"] = zp

            zmx = dvx - dbx_a
            zmy = dvy - dby_a
            zmz = dvz - dbz_a
            zm = np.sqrt(np.mean(zmx**2+zmy**2+zmz**2))
            df.at[i, "zm"] = zm

    except Exception as e:
        print("Error: missingness < 10% but error in computations: {}".format(e))

# Plotting all ACFs to check calculations
for acf in acf_lr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_acfs_lr.png")
plt.close()

for acf in velocity_acf_lr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_proton_acfs_lr.png")
plt.close()

for acf in acf_hr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_acfs_hr.png")
plt.close()

df = df.set_index("Timestamp")
df = df.sort_index()

print("\n FINAL DATAFRAME:\n")
print(df.head())

df.to_pickle("data/processed/dataset.pkl")

print("FINISHED")
print("##################################")
