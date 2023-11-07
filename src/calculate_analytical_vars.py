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
import glob

# Custom modules
import params
import utils

file_paths = sorted(glob.iglob("data/processed/dataset_*.pkl"))

print("MERGING NON-OMNI DATASETS")
df_merged = pd.DataFrame({})
for file in file_paths:
    print("Reading " + file)
    df_merged = pd.concat([df_merged, pd.read_pickle(file)])
    #os.remove(file)
df_merged = df_merged.sort_index()

# Dealing with some occasional duplication of timestamps due to a timestamp at the end of a file
# also appearing at the start of the next file 
df_merged = df_merged.groupby(df_merged.index).agg(sum)
# Dealing with any resultant 0s from summing to NAs together
df_merged = df_merged.replace(0, np.nan)

# Bringing in omni data. This needs to be brought in separately as its files are monthly in size, rather than daily
# Also, we do not calculate any secondary variables from the OMNI variables, so we do not need to do this in calculate_numerical_vars.py

omni_file_paths = sorted(glob.iglob("data/processed/" + params.omni_path + params.int_size + "_*.pkl"))
electron_file_paths = sorted(glob.iglob("data/processed/" + params.electron_path + params.int_size + "_*.pkl"))
proton_file_paths = sorted(glob.iglob("data/processed/" + params.proton_path + params.int_size + "_*.pkl"))

print("MERGING OTHER DATASETS")

df_omni = pd.DataFrame({})
for file in omni_file_paths:
    print("Reading " + file)
    df_omni = pd.concat([df_omni, pd.read_pickle(file)])
    #os.remove(omni_file)

df_omni = df_omni.rename(
    columns={
        params.vsw: 'vsw',
        params.p: 'pomni',
        params.Bomni: 'Bomni'})

# Electron data (already have proton data in df_merged)

df_electrons = pd.DataFrame({})
for file in electron_file_paths:
    print("Reading " + file)
    df_electrons = pd.concat([df_electrons, pd.read_pickle(file)])
    #os.remove(omni_file)

df_electrons = df_electrons.rename(
    columns={
        params.ne: 'ne',
        params.Te: 'Te'
    })


# Sunspot data

df_ss = pd.read_csv("data/processed/sunspot_dataset.csv")
df_ss["Timestamp"] = pd.to_datetime(df_ss["Timestamp"])
df_ss.set_index("Timestamp", inplace=True)
# Limit to only the sunspot number column
df_ss = df_ss['SN']
# Limit to only the range of other data
df_ss = df_ss[df_merged.index.min():df_merged.index.max()]
df_ss = df_ss.resample("12H").agg("ffill") # Up-sampling to twice daily

# Merging datasets

print("\nSAVING FULL MERGED DATASET AND SUMMARY STATS TABLE\n")

df_vars = utils.join_dataframes_on_timestamp(df_electrons, df_ss)
df_vars = utils.join_dataframes_on_timestamp(df_vars, df_omni) # Remove if not using OMNI data
df_final = utils.join_dataframes_on_timestamp(df_merged, df_vars)
df_final = df_final.sort_index()

if df_final.index.has_duplicates:
    print("Warning! Final dataframe has duplicate values of the index")

# Calculating analytically-derived variables
# (using ne due to issues with wind ni data)

df_final["rhoe"] = 2.38*np.sqrt(df_final['Te'])/df_final['B0']    
df_final['rhoi'] = 102*np.sqrt(df_final['Tp'])/df_final['B0']
df_final["de"] = 5.31*np.sqrt(df_final["ne"]) # Electron inertial length
df_final["di"] = 228*np.sqrt(df_final["ne"]) # Ion inertial length (swapped ni for ne)
df_final["betae"] = 0.403*df_final["ne"]*df_final["Te"]/df_final["B0"]
df_final["betai"] = 0.403*df_final["np"]*df_final["Tp"]/df_final["B0"]
df_final["va"] = 21.8*df_final['B0']/np.sqrt(df_final["ne"]) # Alfven speed
df_final["ma"] = df_final["V0"]/df_final["va"] # Alfven mach number
df_final["ld"] = 0.00743*np.sqrt(df_final["Te"])/np.sqrt(df_final["ne"]) # Debye length
df_final["p"] = (2e-6)*df_final["np"]*df_final["V0"]**2 # Dynamic pressure in nPa, from https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html

# Calculating Reynolds numbers
df_final["Re_lt"] = (df_final["tcf"]/df_final["ttc"])**2
df_final["Re_di"] = ((df_final["tcf"]*df_final["V0"])/df_final["di"])**(4/3)
df_final["tb"] = 1/((2*np.pi)*df_final["fb"])
df_final["Re_tb"] = ((df_final["tcf"]/df_final["tb"]))**(4/3)

# Converting scales from time to distance
# (invoking Taylor's hypothesis)

df_final['lambda_t_raw'] = df_final["ttu"]*df_final["V0"]
df_final['lambda_t'] = df_final["ttc"]*df_final["V0"]
df_final['lambda_c_e'] = df_final["tce"]*df_final["V0"]
df_final['lambda_c_fit'] = df_final["tcf"]*df_final["V0"]
df_final['lambda_c_int'] = df_final["tci"]*df_final["V0"]

# Elsasser var decay rates
df_final["zp_decay"] = (df_final["zp"]**3)/(df_final["lambda_c_fit"]) # Energy decay/cascade rate
df_final["zn_decay"] = (df_final["zm"]**3)/(df_final["lambda_c_fit"]) # Energy decay/cascade rate

stats = df_final.describe()
print(df_final.info())

df_final = df_final.reset_index() # So that Timestamp is a normal column in the CSV
df_final.to_csv("wind_omni_dataset.csv", index=False)
stats.to_csv("wind_summary_stats.csv")

print("\nChecking for missing data:")
print(df_final.isna().sum()/len(df_final))
print("\n")
print("##############################################")
