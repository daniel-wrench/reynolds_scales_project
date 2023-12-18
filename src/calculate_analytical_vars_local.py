"""
calculate_analytical_vars.py

Processes solar wind data by merging various datasets, calculating analytically-derived 
variables, and saving the results to CSV files.

Custom modules:
    - params: Module containing parameter settings and constants.
    - utils: Utility functions for operations like merging dataframes based on timestamps.

Workflow:
    1. Load datasets:
        - MFI data
        - OMNI data
        - Electron data
        - Proton data
        - Sunspot data
    2. Rename, sort, and preprocess datasets.
    3. Merge datasets based on timestamps.
    4. Calculate derived variables such as Electron gyroradius, Ion gyroradius, 
       Electron inertial length, Ion inertial length, Electron plasma beta, Alfven speed, 
       and Debye length.
    5. Calculate Reynolds numbers and apply Taylor's hypothesis to convert scales from 
       time to distance.
    6. Save the final merged dataset and summary statistics to CSV files.

"""

import pandas as pd
import numpy as np
# Custom modules
# Add src. prefix to below when running interactively
import params
import utils

# Bringing in output from calculate_numerical_vars.py (magnetic and velocity data)

df_num = pd.read_pickle("data/processed/dataset.pkl")
df_num = df_num.sort_index()

# Bringing in OMNI data

df_omni = pd.read_pickle("data/processed/" + params.omni_path + params.int_size + ".pkl")
df_omni = df_omni.rename(
    columns={
        params.vsw: 'Vomni',
        params.p: 'pomni',
        params.Bomni: 'Bomni'})

# Bringing in electron data 

df_electrons = pd.read_pickle("data/processed/" + params.electron_path + params.int_size + ".pkl")
df_electrons = df_electrons.rename(
    columns={
        params.ne: 'ne',
        params.Te: 'Te'
    })

# Bringing in sunspot data

df_ss = pd.read_csv("data/processed/sunspot_dataset.csv")
df_ss["Timestamp"] = pd.to_datetime(df_ss["Timestamp"])
df_ss.set_index("Timestamp", inplace=True)
# Limit to only the sunspot number column
df_ss = df_ss['SN']
# Limit to only the range of other data
df_ss = df_ss[df_num.index.min():df_num.index.max()]
df_ss = df_ss.resample("12H").agg("ffill") # Up-sampling to twice daily

# Merging datasets into complete dataframe, df

df = utils.join_dataframes_on_timestamp(df_electrons, df_ss)
df = utils.join_dataframes_on_timestamp(df, df_omni) # Remove if not using OMNI data
df = utils.join_dataframes_on_timestamp(df, df_num)
df = df.sort_index()

if df.index.has_duplicates:
    print("Warning! Final dataframe has duplicate values of the index")

# Calculating analytically-derived variables
# using ne in place of np due to data availability and very small np values, e.g. March-June 2012

df["rhoe"] = 2.38*np.sqrt(df['Te'])/df['B0'] # Electron gyroradius    
df['rhop'] = 102*np.sqrt(df['Tp'])/df['B0'] # Ion gyroradius
df["de"] = 5.31/np.sqrt(df["ne"]) # Electron inertial length
df["dp"] = 228/np.sqrt(df["ne"]) # Ion inertial length
df["betae"] = 0.403*df["ne"]*df["Te"]/(df["B0"]**2) # Electron plasma beta
df["betap"] = 0.403*df["ne"]*df["Tp"]/(df["B0"]**2) # Ion plasma beta
df["vte"] = 419*np.sqrt(df["Te"]) # Electron thermal velocity
df["vtp"] = 9.79*np.sqrt(df["Tp"]) # Ion thermal velocity
df["ms"] = df["V0"]/df["vtp"] # Sonic mach number (total speed)
df["mst"] = df["dv"]/df["vtp"] # Sonic mach number (fluctuations)
df["va"] = 21.8*df['B0']/np.sqrt(df["ne"]) # Alfven speed
df["ma"] = df["V0"]/df["va"] # Alfven mach number (total speed)
df["mat"] = df["dv"]/df["va"] # Alfven mach number (fluctuations)
df["ld"] = 0.00743*np.sqrt(df["Te"])/np.sqrt(df["ne"]) # Debye length
df["p"] = (2e-6)*df["ne"]*df["V0"]**2 # Proton ram pressure in nPa, from https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html
df["fce"] = 28*df["B0"] # Electron gyrofrequency
df["fci"] = 0.0152*df["B0"] # Ion gyrofrequency

# Calculating Reynolds numbers (using pre-factors derived in paper)
df["Re_lt"] = 27*(df["tcf"]/df["ttc"])**2
df["Re_di"] = 2*((df["tcf"]*df["V0"])/df["dp"])**(4/3)
df["tb"] = 1/((2*np.pi)*df["fb"])
df["Re_tb"] = 2*((df["tcf"]/df["tb"]))**(4/3)

# Converting scales from time to distance
# (invoking Taylor's hypothesis)

df['lambda_t_raw'] = df["ttu"]*df["V0"]
df['lambda_t'] = df["ttc"]*df["V0"]
df['lambda_c_e'] = df["tce"]*df["V0"]
df['lambda_c_fit'] = df["tcf"]*df["V0"]
df['lambda_c_int'] = df["tci"]*df["V0"]

# Elsasser var decay rates
df["zp_decay"] = (df["zp"]**3)/(df["lambda_c_fit"]) # Energy decay/cascade rate
df["zm_decay"] = (df["zm"]**3)/(df["lambda_c_fit"]) # Energy decay/cascade rate

# Rearrange columns
df = df[[
    'missing_mfi', 
    'missing_3dp', 
    'SN', 
    'ma', 
    'mat', 
    'ms', 
    'mst', 
    'betae', 
    'betap', 
    'sigma_c', 
    'sigma_c_abs',
    'sigma_r', 
    'ra', 
    'cos_a', 
    'qi', 
    'qk', 
    'Re_lt', 
    'Re_di', 
    'Re_tb',
    'B0', 
    'Bomni', 
    'db', 
    'dboB0',
    'ne', 
    'np', 
    'nalpha', 
    'Te', 
    'Tp', 
    'Talpha', 
    'rhoe', 
    'rhop', 
    'de', 
    'dp',
    'ld', 
    'lambda_c_fit', 
    'lambda_c_e', 
    'lambda_c_int',
    'lambda_t_raw', 
    'lambda_t',
    'tcf', 
    'tce', 
    'tce_velocity',
    'tci', 
    'ttu', 
    'ttu_std', 
    'ttc', 
    'ttc_std', 
    'tb', 
    'fb',    
    'fce',
    'fci',
    'V0', 
    'v_r', 
    'Vomni', 
    'dv', 
    'va', 
    'db_a', 
    'vte', 
    'vtp', 
    'zp', 
    'zm', 
    'zp_decay', 
    'zm_decay', 
    'p', 
    'pomni', 
    ]]

stats = df.describe()
print(df.info())

df = df.reset_index() # So that Timestamp is a normal (first) column in the CSV
df.to_csv("wind_omni_dataset_WEEK.csv", index=False) # Named so as should be only subset of the full dataset
stats.to_csv("wind_summary_stats_WEEK.csv")

print("\nChecking for missing data:")
print(df.isna().sum()/len(df))
print("\n")
print("##############################################")
