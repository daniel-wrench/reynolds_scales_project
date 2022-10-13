
import params
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("#######################################")
print("PROCESSING DATA FOR SOLAR WIND DATABASE")
print("#######################################")

# In terms of intermediate output for checking, the most important would be the high-res and low-res mag
# field stuff, given this is not retained in the final database

print("\nREADING PICKLE FILES")

# Omni data

df_omni = pd.read_pickle("data/processed/" + params.omni_path + params.int_size + ".pkl")
df_omni = df_omni.rename(
    columns={
        params.vsw: 'vsw',
        params.p: 'p',
        params.Bomni: 'Bomni'})

# Electron data

df_electrons = pd.read_pickle("data/processed/" + params.electron_path + params.int_size + ".pkl")
df_electrons = df_electrons.rename(
    columns={
        params.ne: 'ne',
        params.Te: 'Te'
    })


# Proton data

df_protons = pd.read_pickle("data/processed/" + params.proton_path + params.int_size + ".pkl")
df_protons = df_protons.rename(
    columns={
        params.ni: 'ni',
        params.Ti: 'Ti'})


df = utils.join_dataframes_on_timestamp(df_omni, df_electrons)
df = utils.join_dataframes_on_timestamp(df, df_protons)

## Wind magnetic field data

# High-res data

df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + ".pkl")
df_wind_hr = df_wind_hr.rename(
    columns={
        params.Bwind: 'Bwind',
        params.Bx: 'Bx',
        params.By: 'By',
        params.Bz: 'Bz'})

print("\nHigh-res Wind dataframe:\n")
print(df_wind_hr.info())
print(df_wind_hr.describe().round(2))

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

df_wind_lr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_lr + ".pkl")

df_wind_lr = df_wind_lr.rename(
    columns={
        params.Bwind: 'Bwind',
        params.Bx: 'Bx',
        params.By: 'By',
        params.Bz: 'Bz'})

print("\nLow-res Wind dataframe:\n")
print(df_wind_lr.info())
print(df_wind_lr.describe().round(2))

# Splitting entire dataframe into a list of intervals

wind_df_hr_list = []
wind_df_lr_list = []

starttime = df_wind_lr.index[0] # E.g. 2016-01-01 00:00
endtime = starttime + pd.to_timedelta(params.int_size) - pd.to_timedelta("0.01S") # E.g. 2016-01-01 11:59:59.99

n_int = np.round((df_wind_lr.index[-1]-df_wind_lr.index[0]) /
                 pd.to_timedelta(params.int_size)).astype(int)

for i in np.arange(n_int).tolist():
    wind_df_lr_list.append(df_wind_lr[(starttime + i*pd.to_timedelta(params.int_size)):(endtime + i*pd.to_timedelta(params.int_size))])
    wind_df_hr_list.append(df_wind_hr[(starttime + i*pd.to_timedelta(params.int_size)):(endtime + i*pd.to_timedelta(params.int_size))])

print("\n\nNumber of high-res Wind intervals = {}".format(len(wind_df_hr_list)))
print("First high-res Wind interval:\n")
print(wind_df_hr_list[0].info())

print("\n\nNumber of low-res Wind intervals = {}".format(len(wind_df_lr_list)))
print("First low-res Wind interval:\n")
print(wind_df_lr_list[0].info())

# Computing ACFs for each low-res interval

acf_lr_list = []

print("\nCOMPUTING LOW-RES ACFS")

for i in np.arange(len(wind_df_lr_list)):

    time_lags_lr, acf = utils.compute_nd_acf(
        np.array([wind_df_lr_list[i].Bx, wind_df_lr_list[i].By,
                 wind_df_lr_list[i].Bz]),
        nlags=params.nlags_lr,
        dt=float(params.dt_lr[:-1]))  # Removing "S" from end

    acf_lr_list.append(acf)

for acf in acf_lr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_acfs_lr.png")
plt.close()

# Computing ACFs and spectral statistics for each high-res interval
# ~1min per interval due to spectrum smoothing algorithm

acf_hr_list = []
inertial_slope_list = []
kinetic_slope_list = []
spectral_break_list = []

print("\nCOMPUTING HIGH-RES ACFS, SPECTRA")

for i in np.arange(len(wind_df_hr_list)):

    time_lags_hr, acf = utils.compute_nd_acf(
        np.array([
            wind_df_hr_list[i].Bx,
            wind_df_hr_list[i].By,
            wind_df_hr_list[i].Bz
        ]),
        nlags=params.nlags_hr,
        dt=float(params.dt_hr[:-1]))

    acf_hr_list.append(acf)

    slope_i, slope_k, break_s = utils.compute_spectral_stats(
        np.array([
            wind_df_hr_list[i].Bx,
            wind_df_hr_list[i].By,
            wind_df_hr_list[i].Bz
        ]),
        dt=float(params.dt_hr[:-1]),
        f_min_inertial=params.f_min_inertial, f_max_inertial=params.f_max_inertial,
        f_min_kinetic=params.f_min_kinetic, f_max_kinetic=params.f_max_kinetic,
        show=False)

    inertial_slope_list.append(slope_i)
    kinetic_slope_list.append(slope_k)
    spectral_break_list.append(break_s)

for acf in acf_hr_list:
    plt.plot(acf)
plt.savefig("data/processed/all_acfs_hr.png")
plt.close()

# Computing scales for each interval

corr_scale_exp_fit_list = []
corr_scale_exp_trick_list = []
corr_scale_int_list = []

print("\nCOMPUTING CORRELATION SCALES")

for acf in acf_lr_list:

    corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(
        time_lags_lr, acf)
    corr_scale_exp_trick_list.append(corr_scale_exp_trick)

    # Use estimate from 1/e method to select fit amount
    corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(
        time_lags_lr, acf, np.round(2*corr_scale_exp_trick), show=False)
    corr_scale_exp_fit_list.append(corr_scale_exp_fit)

    corr_scale_int = utils.compute_outer_scale_integral(time_lags_lr, acf)
    corr_scale_int_list.append(corr_scale_int)

# Re-calculating Kevin's values for checking against his results.
# Also returning both corrected and un-corrected Chuychai versions
taylor_scale_kevin_list = []

taylor_scale_u_list = []
taylor_scale_u_std_list = []

taylor_scale_c_list = []
taylor_scale_c_std_list = []

print("\nCOMPUTING TAYLOR SCALES")

for i in range(len(acf_hr_list)):

    taylor_scale_kevin = utils.compute_taylor_scale(
        time_lags_hr,
        acf_hr_list[i],
        tau_fit=20)

    taylor_scale_kevin_list.append(taylor_scale_kevin)

    taylor_scale_u, taylor_scale_u_std = utils.compute_taylor_chuychai(
        time_lags=time_lags_hr,
        acf=acf_hr_list[i],
        tau_min=params.tau_min,
        tau_max=params.tau_max)

    taylor_scale_u_list.append(taylor_scale_u)
    taylor_scale_u_std_list.append(taylor_scale_u_std)

    taylor_scale_c, taylor_scale_c_std = utils.compute_taylor_chuychai(
        time_lags=time_lags_hr,
        acf=acf_hr_list[i],
        tau_min=params.tau_min,
        tau_max=params.tau_max,
        q=kinetic_slope_list[i])

    taylor_scale_c_list.append(taylor_scale_c)
    taylor_scale_c_std_list.append(taylor_scale_c_std)


# Joining lists of scales and spectral_stats together into a dataframe

df_1 = pd.DataFrame({
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

# Joining all data together into a dataframe
df_5 = df.reset_index()
df_complete = df_5.join(df_1)
stats = df_complete.describe()

print("\nSAVING FINAL DATASET AND SUMMARY STATS TABLE\n")
print(df_complete.info())

# Saving final dataframe and summary stats
df_complete.to_csv("data/processed/db_wind.csv", index=False)
stats.to_csv("data/processed/db_wind_summary_stats.csv")

# Outputting some plots of the ACF and fitting for extreme and middle values of each scale
# Can use to valuate the current settings of these numerical methods

print("SAVING SETTING-EVALUATION PLOTS")

# Smallest tcf
utils.compute_outer_scale_exp_fit(
    time_lags=time_lags_lr,
    acf=acf_lr_list[df_complete.index[df_complete["tcf"]
                                      == df_complete["tcf"].min()][0]],
    seconds_to_fit=np.round(
        2*df_complete.loc[df_complete["tcf"] == df_complete["tcf"].min(), "tce"]),
    save=True,
    figname="tcf_smallest")

# ~ Median tcf
median_ish = df_complete.sort_values("tcf").reset_index()[
    "tcf"][round(len(acf_hr_list)/2)]  # Fix index

utils.compute_outer_scale_exp_fit(
    time_lags=time_lags_lr,
    acf=acf_lr_list[df_complete.index[df_complete["tcf"] == median_ish][0]],
    seconds_to_fit=np.round(
        2*df_complete.loc[df_complete["tcf"] == median_ish, "tce"]),
    save=True,
    figname="tcf_median")

# Largest tcf
utils.compute_outer_scale_exp_fit(
    time_lags=time_lags_lr,
    acf=acf_lr_list[df_complete.index[df_complete["tcf"]
                                      == df_complete["tcf"].max()][0]],
    seconds_to_fit=np.round(
        2*df_complete.loc[df_complete["tcf"] == df_complete["tcf"].max(), "tce"]),
    save=True,
    figname="tcf_largest")

# Smallest ttc acf
plt.plot(time_lags_hr,
         acf_hr_list[df_complete.index[df_complete["ttc"] == df_complete["ttc"].min()][0]])
plt.title("ttc_smallest_acf")
plt.savefig("data/processed/ttc_smallest_acf.png", bbox_inches='tight')
plt.close()

# Smallest ttc fitting
utils.compute_taylor_chuychai(
    time_lags=time_lags_hr,
    acf=acf_hr_list[df_complete.index[df_complete["ttc"]
                                      == df_complete["ttc"].min()][0]],
    tau_min=10,
    tau_max=50,
    q=kinetic_slope_list[df_complete.index[df_complete["ttc"]
                                           == df_complete["ttc"].min()][0]],
    save=True,
    figname="ttc_smallest")

# ~ Median ttc acf
median_ish = df_complete.sort_values("ttc").reset_index()[
    "ttc"][round(len(acf_hr_list)/2)]

plt.plot(time_lags_hr,
         acf_hr_list[df_complete.index[df_complete["ttc"] == median_ish][0]])
plt.title("median_ttc_acf")
plt.savefig("data/processed/ttc_median_acf.png", bbox_inches='tight')
plt.close()

# ~ Median ttc fitting
utils.compute_taylor_chuychai(
    time_lags=time_lags_hr,
    acf=acf_hr_list[df_complete.index[df_complete["ttc"] == median_ish][0]],
    tau_min=10,
    tau_max=50,
    q=kinetic_slope_list[df_complete.index[df_complete["ttc"]
                                           == median_ish][0]],
    save=True,
    figname="ttc_median")

# Largest ttc acf
plt.plot(time_lags_hr,
         acf_hr_list[df_complete.index[df_complete["ttc"] == df_complete["ttc"].max()][0]])
plt.title("largest_ttc_acf")
plt.savefig("data/processed/ttc_largest_acf.png", bbox_inches='tight')
plt.close()

# Largest ttc fitting
utils.compute_taylor_chuychai(
    time_lags=time_lags_hr,
    acf=acf_hr_list[df_complete.index[df_complete["ttc"]
                                      == df_complete["ttc"].max()][0]],
    tau_min=10,
    tau_max=50,
    q=kinetic_slope_list[df_complete.index[df_complete["ttc"]
                                           == df_complete["ttc"].max()][0]],
    save=True,
    figname="ttc_largest")

print("\nFINISHED")
print("##################################")
