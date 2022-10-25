
import params
import utils
import numpy as np
import pandas as pd

####### PARALLEL STUFF #######
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()
##############################

if rank == 0:
    print("#######################################")
    print("PROCESSING DATA FOR SOLAR WIND DATABASE")
    print("#######################################")

comm.Barrier()
# In terms of intermediate output for checking, the most important would be the high-res and low-res mag
# field stuff, given this is not retained in the final database

print("\nCORE {}: READING PICKLE FILES".format(rank))

# Electron data

df_electrons = pd.read_pickle("data/processed/" + params.electron_path + params.int_size + "_{:03d}.pkl".format(rank))
df_electrons = df_electrons.rename(
    columns={
        params.ne: 'ne',
        params.Te: 'Te'
    })

# Proton data

df_protons = pd.read_pickle("data/processed/" + params.proton_path + params.int_size + "_{:03d}.pkl".format(rank))
df_protons = df_protons.rename(
    columns={
        params.ni: 'ni',
        params.Ti: 'Ti'})


df = utils.join_dataframes_on_timestamp(df_electrons, df_protons)

## Wind magnetic field data

# High-res data

df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + "_{:03d}.pkl".format(rank))
df_wind_hr = df_wind_hr.rename(
    columns={
        params.Bwind: 'Bwind',
        params.Bx: 'Bx',
        params.By: 'By',
        params.Bz: 'Bz'})

# Adding magnetic field fluctuations (just going as far as calculating db for now)

dbx = df_wind_hr["Bx"] - df_wind_hr["Bx"].mean()
dby = df_wind_hr["By"] - df_wind_hr["By"].mean()
dbz = df_wind_hr["Bz"] - df_wind_hr["Bz"].mean()
db = np.sqrt(dbx**2+dby**2+dbz**2).rename("db")

#B0 = np.sqrt(df_wind_hr["BGSE_0"].mean()**2 + df_wind_hr["BGSE_1"].mean()**2 + df_wind_hr["BGSE_2"].mean()**2)
#dboB0 = db/B0

# Taking the mean for interval to add as a column to the final dataframe, then dropping these columns from the original df

turb_fluc_hr = db.resample(params.int_size).mean()
b0_hr = df_wind_hr["Bwind"].resample(params.int_size).mean()

df = utils.join_dataframes_on_timestamp(df, turb_fluc_hr)
df = utils.join_dataframes_on_timestamp(df, b0_hr)

# Calculating analytically-derived variables

df["rhoe"] = (2.38e-5)*(df["Te"]**(1/2))*((df["Bwind"]*1e-5)**-1)  # Electron gyroradius
df["rhoi"] = (1.02e-3)*(df["Ti"]**(1/2))*((df["Bwind"]*1e-5)**-1) # Ion gyroradius
df["de"] = (5.31)*(df["ne"]**(-1/2)) # Electron inertial length
df["di"] = (2.28e2)*(df["ni"]**(-1/2)) # Ion inertial length
df["betae"] = (4.03e-11)*df["ne"]*df["Te"]*((df["Bwind"]*1e-5)**-2) # Electron plasma beta
df["betai"] = (4.03e-11)*df["ni"]*df["Ti"]*((df["Bwind"]*1e-5)**-2) # Ion plasma beta
df["va"] = (2.18e6)*(df["ni"]**(-1/2))*(df["Bwind"]*1e-5) # Alfven speed
df["ld"] = (7.43e-3)*(df["Te"]**(1/2))*(df["ne"]**(-1/2)) # Debye length

# Low-res data

df_wind_lr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_lr + "_{:03d}.pkl".format(rank))

df_wind_lr = df_wind_lr.rename(
    columns={
        params.Bwind: 'Bwind',
        params.Bx: 'Bx',
        params.By: 'By',
        params.Bz: 'Bz'})

inertial_slope_list = []
kinetic_slope_list = []
spectral_break_list = []
corr_scale_exp_fit_list = []
corr_scale_exp_trick_list = []
corr_scale_int_list = []

# Re-calculating Kevin's values for checking against his results.
# Also returning both corrected and un-corrected Chuychai versions
taylor_scale_kevin_list = []

taylor_scale_u_list = []
taylor_scale_u_std_list = []

taylor_scale_c_list = []
taylor_scale_c_std_list = []
# Splitting entire dataframe into a list of intervals

wind_df_hr_list_missing = [] # This will be added as a column to the final dataframe


starttime = df_wind_lr.index[0] # E.g. 2016-01-01 00:00
endtime = starttime + pd.to_timedelta(params.int_size) - pd.to_timedelta("0.01S") # E.g. 2016-01-01 11:59:59.99

n_int = np.round((df_wind_lr.index[-1]-df_wind_lr.index[0]) /
                 pd.to_timedelta(params.int_size)).astype(int)

for i in np.arange(n_int).tolist():
    int_lr = df_wind_lr[(starttime + i*pd.to_timedelta(params.int_size)):(endtime + i*pd.to_timedelta(params.int_size))]
    int_hr = df_wind_hr[(starttime + i*pd.to_timedelta(params.int_size)):(endtime + i*pd.to_timedelta(params.int_size))]

    if int_hr.empty:
        missing = 1
    else:
        missing =  int_hr.iloc[:,0].isna().sum()/len(int_hr)

    wind_df_hr_list_missing.append(missing)

    if missing > 0.1:
        inertial_slope_list.append(np.nan)
        kinetic_slope_list.append(np.nan)
        spectral_break_list.append(np.nan)
        corr_scale_exp_trick_list.append(np.nan)
        corr_scale_exp_fit_list.append(np.nan)
        corr_scale_int_list.append(np.nan)
        taylor_scale_kevin_list.append(np.nan)
        taylor_scale_u_list.append(np.nan)
        taylor_scale_u_std_list.append(np.nan)
        taylor_scale_c_list.append(np.nan)
        taylor_scale_c_std_list.append(np.nan)
    else:
        try:
            int_hr = int_hr.interpolate().ffill().bfill()
            int_lr = int_lr.interpolate().ffill().bfill()

            time_lags_lr, acf_lr = utils.compute_nd_acf(
                np.array([
                    int_lr.Bx, 
                    int_lr.By,
                    int_lr.Bz
                ]),
                nlags=params.nlags_lr,
                dt=float(params.dt_lr[:-1]))  # Removing "S" from end

            corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(time_lags_lr, acf_lr)
            corr_scale_exp_trick_list.append(corr_scale_exp_trick)

            # Use estimate from 1/e method to select fit amount
            corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(
                time_lags_lr, acf_lr, np.round(2*corr_scale_exp_trick), show=False)
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

        # ~1min per interval due to spectrum smoothing algorithm
            slope_i, slope_k, break_s = utils.compute_spectral_stats(
                np.array([
                    int_hr.Bx,
                    int_hr.By,
                    int_hr.Bz
                ]),
                dt=float(params.dt_hr[:-1]),
                f_min_inertial=params.f_min_inertial, f_max_inertial=params.f_max_inertial,
                f_min_kinetic=params.f_min_kinetic, f_max_kinetic=params.f_max_kinetic,
                show=False)

            inertial_slope_list.append(slope_i)
            kinetic_slope_list.append(slope_k)
            spectral_break_list.append(break_s)

            taylor_scale_kevin = utils.compute_taylor_scale(
                time_lags_hr,
                acf_hr,
                tau_fit=20)

            taylor_scale_kevin_list.append(taylor_scale_kevin)

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
        except:
            print("Error: missingness < 0.4 but error in computations")


# Joining lists of scales and spectral_stats together into a dataframe

df_1 = pd.DataFrame({
    'missing_mfi': wind_df_hr_list_missing,
    'tcf': corr_scale_exp_fit_list,
    'tce': corr_scale_exp_trick_list,
    'tci': corr_scale_int_list,
    'ttu': taylor_scale_u_list,
    'ttu_std': taylor_scale_u_std_list,
    'ttc': taylor_scale_c_list,
    'ttc_std': taylor_scale_c_std_list,
    'ttk': taylor_scale_kevin_list,
    'qi': inertial_slope_list,
    'qk': kinetic_slope_list,
    'tb': spectral_break_list
})

print("\nCORE {}: JOINING COLUMNS INTO SINGLE DATAFRAME".format(rank))
df_5 = df.reset_index()
df_complete = df_5.join(df_1)
df_final = df_complete.set_index("Timestamp")

df_final.to_pickle("data/processed/dataset_{:03d}.pkl".format(rank))

print("\nCORE {}: FINISHED".format(rank))

comm.Barrier()

if rank == 0:
    print("##################################")
