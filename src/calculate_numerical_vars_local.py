
"""
calculate_numerical_vars.py

This script processes high-resolution and low-resolution wind magnetic field data to compute various metrics related to magnetic field fluctuations and their statistical properties.

Modules:
    - params: Contains parameters and configurations for data processing.
    - utils: Contains utility functions for data processing and analysis.

Steps:
    1. Read high-resolution and low-resolution wind magnetic field data from pickle files.
    2. Process data and initialize lists for storing computed metrics.
    3. Loop over each interval to compute autocorrelations and power spectra, handling missing data as necessary.
    4. Create a DataFrame containing the computed statistics for each interval.
    5. Save the final processed dataframes as pickle files.

Author: Daniel Wrench
Last modified: 4/9/2023
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Add src. prefix to following if running interactively
import params as params
import utils as utils

# In terms of intermediate output for checking, the most important would be the high-res and low-res mag
# field stuff, given this is not retained in the final database

print("\nREADING PICKLE FILES")

# Wind magnetic field data

## High-res data

df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + ".pkl")

df_wind_hr = df_wind_hr.rename(
    columns={
        params.Bwind: "Bwind",
        params.Bx: "Bx",
        params.By: "By",
        params.Bz: "Bz"})

print("\nHigh-res Wind dataframe:\n")
print(df_wind_hr.info())
print(df_wind_hr.describe().round(2))

# WANT TO LIMIT THE FOLLOWING ANALYSIS TO EACH INTERVAL
# but check it works here first

# We also need velocities for cross-helicity and converting to Alfvenic units

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

############

# Low-res data

df_wind_lr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_lr + ".pkl")

df_wind_lr = df_wind_lr.rename(
    columns={
        params.Bx: "Bx",
        params.By: "By",
        params.Bz: "Bz"})

print("\nLow-res Wind dataframe:\n")
print(df_wind_lr.info())
print(df_wind_lr.describe().round(2))

print("\nFINISHED READING DATA")


timestamps = []

# Initializing lists for storing computed metrics
Bwind_list = []
B0_list = []
dboB0_list = []
V0_list = []
v_r_list = []
dv_list = []
zp_list = []
zm_list = []
sigma_c_list = []
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
corr_scale_int_list = []

# Returning both corrected and un-corrected Chuychai versions

taylor_scale_u_list = []
taylor_scale_u_std_list = []

taylor_scale_c_list = []
taylor_scale_c_std_list = []

# Splitting entire dataframe into a list of intervals

wind_df_hr_list_missing = []  # This will be added as a column to the final dataframe
wind_df_protons_list_missing = []

# Saving acfs to list to then output as plot
acf_hr_list = []
acf_lr_list = []

# E.g. 2016-01-01 00:00
starttime = df_wind_lr.index[0].round(params.int_size)
# Will account for when the dataset does not start at a nice round 12H timestamp

# E.g. 2016-01-01 11:59:59.99
endtime = starttime + pd.to_timedelta(params.int_size) - pd.to_timedelta("0.01S")

n_int = np.round((df_wind_lr.index[-1]-df_wind_lr.index[0]) /
                 pd.to_timedelta(params.int_size)).astype(int)

# If we subset timestamps that don"t exist in the dataframe, they will still be included in the list, just as
# missing dataframes. We can identify these with df.empty = True (or missing)

print("\nLOOPING OVER EACH INTERVAL")

for i in np.arange(n_int).tolist():

    int_start = starttime + i*pd.to_timedelta(params.int_size)
    int_end = (endtime + i*pd.to_timedelta(params.int_size))

    timestamps.append(int_start)

    int_lr = df_wind_lr[int_start:int_end]
    int_hr = df_wind_hr[int_start:int_end]
    int_protons = df_protons[int_start:int_end]

    # Record and deal with missing data
    if int_hr.empty:
        missing_mfi = 1
    else:
        missing_mfi = int_hr.iloc[:, 0].isna().sum()/len(int_hr)
        missing_3dp = int_protons.iloc[:, 0].isna().sum()/len(int_protons)

    wind_df_hr_list_missing.append(missing_mfi)
    wind_df_protons_list_missing.append(missing_3dp)

    if missing_mfi > 0.1:
        # If magnetic field data is sufficiently sparse, set values that require it (basically all) to missing
        Bwind_list.append(np.nan)
        inertial_slope_list.append(np.nan)
        kinetic_slope_list.append(np.nan)
        spectral_break_list.append(np.nan)
        corr_scale_exp_trick_list.append(np.nan)
        corr_scale_exp_fit_list.append(np.nan)
        corr_scale_int_list.append(np.nan)
        taylor_scale_u_list.append(np.nan)
        taylor_scale_u_std_list.append(np.nan)
        taylor_scale_c_list.append(np.nan)
        taylor_scale_c_std_list.append(np.nan)
        np_list.append(np.nan)
        nalpha_list.append(np.nan)
        Tp_list.append(np.nan)
        Talpha_list.append(np.nan)
        B0_list.append(np.nan)
        dboB0_list.append(np.nan)
        V0_list.append(np.nan)
        v_r_list.append(np.nan)
        dv_list.append(np.nan)
        zp_list.append(np.nan)
        zm_list.append(np.nan)
        sigma_c_list.append(np.nan)
        sigma_r_list.append(np.nan)
        ra_list.append(np.nan)
        cos_a_list.append(np.nan)
    
    else:
        # Interpolate missing data, then fill any remaining gaps at start or end with nearest value
        int_hr = int_hr.interpolate(method="linear").ffill().bfill() 
        int_lr = int_lr.interpolate(method="linear").ffill().bfill()

    if missing_3dp > 0.1:
        # If proton data is sufficiently sparse, set values that require it to missing
        np_list.append(np.nan)
        nalpha_list.append(np.nan)
        Tp_list.append(np.nan)
        Talpha_list.append(np.nan)
        B0_list.append(np.nan)
        dboB0_list.append(np.nan)
        V0_list.append(np.nan)
        v_r_list.append(np.nan)
        dv_list.append(np.nan)
        zp_list.append(np.nan)
        zm_list.append(np.nan)
        sigma_c_list.append(np.nan)
        sigma_r_list.append(np.nan)
        ra_list.append(np.nan)
        cos_a_list.append(np.nan)

    else:
        # Interpolate missing data, then fill any remaining gaps at start or end with nearest value
        int_protons = int_protons.interpolate(method="linear").ffill().bfill()

    try: # try statement for error handling; see except statement at end of loop

        ## Calculating magnetic field fluctuations, db/B0
        ## (Same as previous, except now calculating full db/B0 at this step, not just the fluctuations)
        ## Fluctuations are calculated relative to the mean of the specific dataset read in, however large that may b

        # Temps and densities, raw mag magnitude
        Bwind_list.append(int_hr["Bwind"].mean())
        np_list.append(int_protons["np"].mean())
        nalpha_list.append(int_protons["nalpha"].mean())
        Tp_list.append(int_protons["Tp"].mean())
        Talpha_list.append(int_protons["Talpha"].mean())

        # Resampling mag field data to 3s to match velocity data cadence
        ## NB: there is also a 3s cadence version of the mfi data, e.g. as used by Podesta2010, 
        ## but we want the highest res possible for the Taylor scale calculations

        Bx = int_hr["Bx"].resample(params.dt_protons).mean()
        By = int_hr["By"].resample(params.dt_protons).mean()
        Bz = int_hr["Bz"].resample(params.dt_protons).mean()

        Bx_mean = Bx.mean()
        By_mean = By.mean()
        Bz_mean = Bz.mean()

        # Calculate rms magnetic field
        B0 = np.sqrt(np.mean(Bx**2)+np.mean(By**2)+np.mean(Bz**2))
        B0_list.append(B0)

        # Add velocity field fluctuations, dv
        Vx = int_protons["Vx"]
        Vy = int_protons["Vy"]
        Vz = int_protons["Vz"]

        Vx_mean = Vx.mean()
        Vy_mean = Vy.mean()
        Vz_mean = Vz.mean()

        # Save mean radial velocity (should dominate velocity mag)
        v_r_list.append(np.abs(Vx_mean)) # abs() because all vals negative (away from Sun)

        # Calculate velocity magnitude
        V0 = np.sqrt(np.mean(Vx**2)+np.mean(Vy**2)+np.mean(Vz**2))
        V0_list.append(V0)

        # Add magnetic field fluctuations, db
        dbx = Bx - Bx_mean
        dby = By - By_mean
        dbz = Bz - Bz_mean
        db = np.sqrt(dbx**2+dby**2+dbz**2)
        db_rms = np.sqrt(np.mean(dbx**2)+np.mean(dby**2)+np.mean(dbz**2))
        
        dboB0_list.append(db_rms/B0)

        ## in Alfvenic units
        alfven_prefactor = (2.18e1)/np.sqrt(int_protons["np"].mean()) # Converting nT to Gauss and cm/s to km/s
        # Using mean density of interval a la Wang2012ApJ

        dbx_a = dbx*alfven_prefactor
        dby_a = dby*alfven_prefactor
        dbz_a = dbz*alfven_prefactor
        db_a = np.sqrt(dbx_a**2+dby_a**2+dbz_a**2)

        dvx = (Vx - Vx_mean)
        dvy = (Vy - Vy_mean)
        dvz = (Vz - Vz_mean)

        dv = np.sqrt(dvx**2+dvy**2+dvz**2)
        dv_rms = np.sqrt(np.mean(dvx**2)+np.mean(dvy**2)+np.mean(dvz**2))
        dv_list.append(dv_rms)

        # Cross-helicity 
        Hc = np.mean(dvx*dbx_a + dvy*dby_a + dvz*dbz_a)

        # Normalize by energy (should then range between -1 and 1, like a normal correlation coefficient)
        # Only minor different calculating them separately, rather than np.mean(dv**2 + db**2)
        e_kinetic = np.mean(dv**2)
        e_magnetic = np.mean(db_a**2)

        sigma_c = 2*Hc/(e_kinetic+e_magnetic)
        sigma_c_list.append(sigma_c)

        # Normalized residual energy
        sigma_r = (e_kinetic-e_magnetic)/(e_kinetic+e_magnetic)
        sigma_r_list.append(sigma_r)

        # Alfven ratio (ratio between kinetic and magnetic energy)
        ra = e_kinetic/e_magnetic
        ra_list.append(ra)

        # Alignment cosine (see Parashar2018PRL)
        cos_a = Hc/np.mean(np.sqrt(e_kinetic*e_magnetic))
        cos_a_list.append(cos_a)

        # Elsasser variables
        zpx = dvx + dbx_a
        zpy = dvy + dby_a
        zpz = dvz + dbz_a
        zp = np.sqrt(np.mean(zpx**2)+np.mean(zpy**2)+np.mean(zpz**2))
        zp_list.append(zp)

        zmx = dvx - dbx_a
        zmy = dvy - dby_a
        zmz = dvz - dbz_a
        zm = np.sqrt(np.mean(zmx**2)+np.mean(zmy**2)+np.mean(zmz**2))
        zm_list.append(zm)

        # Compute autocorrelations and power spectra
        time_lags_lr, acf_lr = utils.compute_nd_acf(
            np.array([
                int_lr.Bx,
                int_lr.By,
                int_lr.Bz
            ]),
            nlags=params.nlags_lr,
            dt=float(params.dt_lr[:-1]))  # Removing "S" from end of dt string

        acf_lr_list.append(acf_lr)

        corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(time_lags_lr, acf_lr)
        corr_scale_exp_trick_list.append(corr_scale_exp_trick)

        # Use estimate from 1/e method to select fit amount
        corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(
            time_lags_lr, acf_lr, np.round(2*corr_scale_exp_trick))
        corr_scale_exp_fit_list.append(corr_scale_exp_fit)

        corr_scale_int = utils.compute_outer_scale_integral(time_lags_lr, acf_lr)
        corr_scale_int_list.append(corr_scale_int)

        time_lags_hr, acf_hr = utils.compute_nd_acf(
            np.array([
                int_hr.Bx,
                int_hr.By,
                int_hr.Bz
            ]),
            nlags=params.nlags_hr,
            dt=float(params.dt_hr[:-1]))

        acf_hr_list.append(acf_hr)

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

        inertial_slope_list.append(slope_i)
        kinetic_slope_list.append(slope_k)
        spectral_break_list.append(break_s)

        taylor_scale_u, taylor_scale_u_std = utils.compute_taylor_chuychai(
            time_lags_hr,
            acf_hr,
            tau_min=params.tau_min,
            tau_max=params.tau_max)

        taylor_scale_u_list.append(taylor_scale_u)
        taylor_scale_u_std_list.append(taylor_scale_u_std)

        taylor_scale_c, taylor_scale_c_std = utils.compute_taylor_chuychai(
            time_lags_hr,
            acf_hr,
            tau_min=params.tau_min,
            tau_max=params.tau_max,
            q=slope_k)

        taylor_scale_c_list.append(taylor_scale_c)
        taylor_scale_c_std_list.append(taylor_scale_c_std)
    
    except Exception as e:
        print("Error: missingness < 10% but error in computations: {}".format(e))

# Plotting all ACFs to check calculations
for acf in acf_lr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_acfs_lr.png")
plt.close()

for acf in acf_hr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_acfs_hr.png")
plt.close()


# Create a dataframe combining all of the lists above
df = pd.DataFrame({
    "Timestamp": timestamps,
    "missing_mfi": wind_df_hr_list_missing,
    "np": np_list,
    "nalpha": nalpha_list,
    "Tp": Tp_list,
    "Talpha": Talpha_list,
    "B0": B0_list,
    "dboB0": dboB0_list,
    "V0": V0_list,
    "v_r": v_r_list,
    "dv": dv_list,
    "zp": zp_list,
    "zm": zm_list,
    "sigma_c": sigma_c_list,
    "sigma_r": sigma_r_list,
    "ra": ra_list,
    "cos_a": cos_a_list,
    "qi": inertial_slope_list,
    "qk": kinetic_slope_list,
    "fb": spectral_break_list,
    "tcf": corr_scale_exp_fit_list,
    "tce": corr_scale_exp_trick_list,
    "tci": corr_scale_int_list,
    "ttu": taylor_scale_u_list,
    "ttu_std": taylor_scale_u_std_list,
    "ttc": taylor_scale_c_list,
    "ttc_std": taylor_scale_c_std_list
})

df = df.set_index("Timestamp")
df = df.sort_index()

# df.to_csv("week_check.csv")
df.to_pickle("data/processed/dataset.pkl")

print("FINISHED")
print("##################################")
