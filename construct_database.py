
print("##################################")
print("PROCESSING DATA FOR WIND DATABASE")
print("##################################")

# In terms of intermediate output for checking, the most important would be the high-res and low-res mag
# field stuff, given this is not retained in the final database

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from src import utils

# ### Omni data
# 6-hour averages, first processed and saved to `.pkl` file by `process_data_omni.py`

df_omni = pd.read_pickle("data/processed/omni_12hr.pkl")
df_omni = df_omni.rename(
    columns={
        'flow_speed':'vsw',
        'Pressure':'p',
        'F':'Bomni'})

# Demonstrating use of Units features from `astropy` package (not currently used)

# Can also use QTable objet for smarter behaviour

# omni_table = Table.from_pandas(df_omni.reset_index())
# omni_table.add_index("Timestamp")
# omni_table["proton_density_omni"] = omni_table["proton_density_omni"] /u.centimeter**3
# omni_table["temperature"] = omni_table["temperature"] * u.K
# omni_table["flow_speed"] = omni_table["flow_speed"] * u.km /u.s
# omni_table["flow_pressure"] = omni_table["flow_pressure"] * u.nPa
# omni_table["electric_field"] = omni_table["electric_field"] * u.mV / u.m
# omni_table


# ### Wind electron data
# 6-hour averages, first processed and saved to `.pkl` file in `process_data_wind_electrons.py`


df_electrons = pd.read_pickle('data/processed/wi_elm2_3dp_12hr.pkl')
df_electrons.columns = df_electrons.columns.str.lower()
df_electrons = df_electrons.rename(
    columns={
        'density':'ne',
        'avgtemp':'Te'
})


# ### Wind proton data
# 
# 6-hour averages, first processed and saved to `.pkl` file in `process_data_wind_protons.py`

df_protons = pd.read_pickle("data/processed/wi_plsp_3dp_12hr.pkl")
df_protons.columns = df_protons.columns.str.lower()
df_protons = df_protons.rename(
    columns={
        'mom.p.density':'ni',
        'mom.p.avgtemp':'Ti'})


df = utils.join_dataframes_on_timestamp(df_omni, df_electrons)
df = utils.join_dataframes_on_timestamp(df, df_protons)

# ### Wind magnetic field data

large_wind_df_hr = pd.read_pickle("data/processed/wi_h2_mfi_hr.pkl")
large_wind_df_hr = large_wind_df_hr.rename(
    columns={
        'BF1':'Bwind',
        'BGSE_0':'Bx',
        'BGSE_1':'By',
        'BGSE_2':'Bz'})

print("\nHigh-res Wind dataframe:\n")
print(large_wind_df_hr.info())
print(large_wind_df_hr.describe().round(2))

# Adding magnetic field fluctuations (just going as far as calculating $db$ for now)

dbx = large_wind_df_hr["Bx"] - large_wind_df_hr["Bx"].mean()
dby = large_wind_df_hr["By"] - large_wind_df_hr["By"].mean()
dbz = large_wind_df_hr["Bz"] - large_wind_df_hr["Bz"].mean()
db = np.sqrt(dbx**2+dby**2+dbz**2).rename("db")

#B0 = np.sqrt(large_wind_df_hr["BGSE_0"].mean()**2 + large_wind_df_hr["BGSE_1"].mean()**2 + large_wind_df_hr["BGSE_2"].mean()**2)
#dboB0 = db/B0

# Taking the mean for each 12-hour interval to add as a column to the final dataframe, then dropping these columns from the original df

turb_fluc_hr = db.resample("12H").mean()
b0_hr = large_wind_df_hr["Bwind"].resample("12H").mean()

df = utils.join_dataframes_on_timestamp(df, turb_fluc_hr)
df = utils.join_dataframes_on_timestamp(df, b0_hr)

# ## Calculating analytically-derived variables


df["rhoe"] = (2.38e-5)*(df["Te"]**(1/2))*(df["Bwind"]**-1)  # Electron gyroradius
df["rhoi"] = (1.02e-3)*(df["Ti"]**(1/2))*(df["Bwind"]**-1) # Ion gyroradius
df["de"] = (5.31)*(df["ne"]**(-1/2)) # Electron inertial length
df["di"] = (2.28e2)*(df["ni"]**(-1/2)) # Ion inertial length
df["betae"] = (4.03e-16)*df["ne"]*df["Te"]*(df["Bwind"]**-2) # Electron plasma beta
df["betai"] = (4.03e-16)*df["ni"]*df["Ti"]*(df["Bwind"]**-2) # Ion plasma beta
df["va"] = (2.18e6)*(df["ni"]**(-1/2))*df["Bwind"] # Alfven speed
df["ld"] = (7.43e-3)*(df["Te"]**(1/2))*(df["ne"]**(-1/2)) # Debye length


# ### 0.2Hz data

large_wind_df_lr = pd.read_pickle("data/processed/wi_h2_mfi_lr.pkl")
large_wind_df_lr = large_wind_df_lr.rename(
    columns={
        'BF1':'Bwind',
        'BGSE_0':'Bx',
        'BGSE_1':'By',
        'BGSE_2':'Bz'})

print("\nLow-res Wind dataframe:\n")
print(large_wind_df_lr.info())
print(large_wind_df_lr.describe().round(2))

# ## Constructing the final dataframe

# Splitting entire dataframe into a list of 12-hour intervals

wind_df_hr_list = []
wind_df_lr_list = []

start = pd.to_datetime("2016-01-01 00:00")
fin = pd.to_datetime("2016-01-01 11:59:59.99") 

n_int = np.round((large_wind_df_lr.index[-1]-large_wind_df_lr.index[0])/pd.to_timedelta("12H")).astype(int)

for i in np.arange(n_int).tolist():
    wind_df_lr_list.append(large_wind_df_lr[(start + datetime.timedelta(hours=i*12)):(fin + datetime.timedelta(hours=i*12))])
    wind_df_hr_list.append(large_wind_df_hr[(start + datetime.timedelta(hours=i*12)):(fin + datetime.timedelta(hours=i*12))])

print("\n\nNumber of high-res Wind intervals = {}".format(len(wind_df_hr_list)))
print("First high-res Wind interval:\n")
print(wind_df_hr_list[0].info())
print(wind_df_hr_list[0].head())

print("\n\nNumber of low-res Wind intervals = {}".format(len(wind_df_lr_list)))
print("First low-res Wind interval:\n")
print(wind_df_lr_list[0].info())
print(wind_df_lr_list[0].head())

# Computing ACFs for each low-res interval
dt_lr = 5

acf_lr_list = []

for i in np.arange(len(wind_df_lr_list)):

    time_lags_lr, acf = utils.compute_nd_acf(
        np.array([wind_df_lr_list[i].Bx, wind_df_lr_list[i].By, wind_df_lr_list[i].Bz]), 
        nlags = 2000, 
        dt=dt_lr)

    acf_lr_list.append(acf)

for acf in acf_lr_list:
    plt.plot(acf)
plt.show()

# Computing ACFs and spectral statistics for each high-res interval
# ~1min per interval due to spectrum smoothing algorithm

dt_hr = 0.091

acf_hr_list = []
inertial_slope_list = []
kinetic_slope_list = []
spectral_break_list = []

for i in np.arange(3):

    time_lags_hr, acf = utils.compute_nd_acf(
        np.array([
            wind_df_hr_list[i].Bx,
            wind_df_hr_list[i].By,
            wind_df_hr_list[i].Bz
        ]),
        nlags=100,
        dt=dt_hr)

    acf_hr_list.append(acf)

    slope_i, slope_k, break_s = utils.compute_spectral_stats(
        np.array([
            wind_df_hr_list[i].Bx,
            wind_df_hr_list[i].By, 
            wind_df_hr_list[i].Bz
        ]),
        dt=dt_hr,
        f_min_inertial=0.01, f_max_inertial=0.2,
        f_min_kinetic=0.5, f_max_kinetic=2,
        show=False)

    inertial_slope_list.append(slope_i)
    kinetic_slope_list.append(slope_k)
    spectral_break_list.append(break_s)

for acf in acf_hr_list:
   plt.plot(acf)
plt.show()
# Computing scales for each interval

corr_scale_exp_fit_list = []
corr_scale_exp_trick_list = []
corr_scale_int_list = []

for acf in acf_lr_list:

    corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(time_lags_lr, acf)
    corr_scale_exp_trick_list.append(corr_scale_exp_trick)

    # Use estimate from 1/e method to select fit amount
    corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(time_lags_lr, acf, np.round(2*corr_scale_exp_trick), show=False)
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

for i in range(len(acf_hr_list)):

    taylor_scale_kevin = utils.compute_taylor_scale(
        time_lags_hr, 
        acf_hr_list[i], 
        tau_fit=20)

    taylor_scale_kevin_list.append(taylor_scale_kevin)

    taylor_scale_u, taylor_scale_u_std = utils.compute_taylor_chuychai(
        time_lags=time_lags_hr,
        acf=acf_hr_list[i],
        tau_min=10,
        tau_max=50)

    taylor_scale_u_list.append(taylor_scale_u)
    taylor_scale_u_std_list.append(taylor_scale_u_std)

    taylor_scale_c, taylor_scale_c_std = utils.compute_taylor_chuychai(
        time_lags=time_lags_hr,
        acf=acf_hr_list[i],
        tau_min=10,
        tau_max=50,
        q=kinetic_slope_list[i])

    taylor_scale_c_list.append(taylor_scale_c)
    taylor_scale_c_std_list.append(taylor_scale_c_std)


# Joining lists of scales and spectral_stats together into a dataframe

df_1 = pd.DataFrame({
    'tcf': corr_scale_exp_fit_list[:3],
    'tce': corr_scale_exp_trick_list[:3],
    'tci': corr_scale_int_list[:3],
    'ttu': taylor_scale_u_list[:3],
    'ttu_std': taylor_scale_u_std_list[:3],
    'ttc': taylor_scale_c_list[:3],
    'ttc_std': taylor_scale_c_std_list[:3],                                      
    'ttk': taylor_scale_kevin_list[:3],
    'qi': inertial_slope_list[:3],
    'qk': kinetic_slope_list[:3],
    'tb': spectral_break_list[:3]
})

# Joining all data together into a dataframe
df_5 = df.reset_index()
df_complete = df_5.join(df_1)
stats = df_complete.describe()

print("\n\nFinal dataset:\n")
print(df_complete.info())

# Saving final dataframe and summary stats
df_complete.to_csv("data/processed/db_wind.csv", index=False)
stats.to_csv("data/processed/db_wind_summary_stats.csv", index = False)

print("\nFINISHED")
print("##################################")


# Let's use the min, median and max values of the **correlation scale (exp fit method)** to evaluate the current settings of the exponential fit function

# (see notebook)