import pandas as pd
import numpy as np
import params
import utils
import glob

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
        params.p: 'p',
        params.Bomni: 'Bomni'})

# Electron data

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

# Proton data

df_protons = pd.DataFrame({})
for file in proton_file_paths:
    print("Reading " + file)
    df_protons = pd.concat([df_protons, pd.read_pickle(file)])
    #os.remove(omni_file)

df_protons = df_protons.rename(
    columns={
        params.ni: 'ni',
        params.Ti: 'Ti'})

# Sunspot data

df_ss = pd.read_csv("data/processed/sunspot_dataset.csv")
df_ss["Timestamp"] = pd.to_datetime(df_ss["Timestamp"])
df_ss.set_index("Timestamp", inplace=True)
# Limit to only the sunspot number column
df_ss = df_ss['SN']
# Limit to only the range of other data
df_ss = df_ss[df_omni.index.min():df_omni.index.max()]
df_ss = df_ss.resample("12H").agg("ffill") # Up-sampling to twice daily

# Merging datasets

print("\nSAVING FULL MERGED DATASET AND SUMMARY STATS TABLE\n")
df_vars = utils.join_dataframes_on_timestamp(df_omni, df_electrons)
df_vars = utils.join_dataframes_on_timestamp(df_vars, df_protons)
df_vars = utils.join_dataframes_on_timestamp(df_vars, df_ss)

df_final = utils.join_dataframes_on_timestamp(df_merged, df_vars)
df_final = df_final.sort_index()

if df_final.index.has_duplicates:
    print("Warning! Final dataframe has duplicate values of the index")

# Calculating analytically-derived variables
# (using ne due to issues with wind ni data)

df_final["rhoe"] = (2.38e-5)*(df_final["Te"]**(1/2))*((df_final["Bwind"]*1e-5)**-1)  # Electron gyroradius
df_final["rhoi"] = (1.02e-3)*(df_final["Ti"]**(1/2))*((df_final["Bwind"]*1e-5)**-1) # Ion gyroradius
df_final["de"] = (5.31)*(df_final["ne"]**(-1/2)) # Electron inertial length
df_final["di"] = (2.28e2)*(df_final["ne"]**(-1/2)) # Ion inertial length (swapped ni for ne)
df_final["betae"] = (4.03e-11)*df_final["ne"]*df_final["Te"]*((df_final["Bwind"]*1e-5)**-2) # Electron plasma beta
#df_final["betai"] = (4.03e-11)*df_final["ne"]*df_final["Ti"]*((df_final["Bwind"]*1e-5)**-2) # Ion plasma beta (now same as betae)
df_final["va"] = (2.18e6)*(df_final["ne"]**(-1/2))*(df_final["Bwind"]*1e-5) # Alfven speed (swapped ni for ne)
df_final["ld"] = (7.43e-3)*(df_final["Te"]**(1/2))*(df_final["ne"]**(-1/2)) # Debye length

# Calculating Reynolds numbers
df_final["Re_lt"] = (df_final["tcf"]/df_final["ttc"])**2
df_final["Re_di"] = ((df_final["tcf"]*df_final["vsw"])/df_final["di"])**(4/3)
df_final["tb"] = 1/((2*np.pi)*df_final["fb"])
df_final["Re_tb"] = ((df_final["tcf"]/df_final["tb"]))**(4/3)

# Converting scales from time to distance
# (invoking Taylor's hypothesis)

df_final['lambda_t_raw'] = df_final["ttu"]*df_final["vsw"]
df_final['lambda_t'] = df_final["ttc"]*df_final["vsw"]
df_final['lambda_c_e'] = df_final["tce"]*df_final["vsw"]
df_final['lambda_c_fit'] = df_final["tcf"]*df_final["vsw"]
df_final['lambda_c_int'] = df_final["tci"]*df_final["vsw"]

stats = df_final.describe()
print(df_final.info())

df_final = df_final.reset_index() # So that Timestamp is a normal column in the CSV
df_final.to_csv("data/processed/wind_omni_dataset.csv", index=False)
stats.to_csv("data/processed/wind_summary_stats.csv")

print("\nChecking for missing data:")
print(df_final.isna().sum()/len(df_final))
print("\n")
print("##############################################")
