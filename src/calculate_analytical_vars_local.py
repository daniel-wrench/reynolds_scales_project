import pandas as pd
import params
import utils
import numpy as np

df_merged = pd.read_pickle("data/processed/dataset.pkl")
df_merged = df_merged.sort_index()

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

print("\nSAVING FULL MERGED DATASET AND SUMMARY STATS TABLE\n")
df_vars = utils.join_dataframes_on_timestamp(df_omni, df_electrons)
df_vars = utils.join_dataframes_on_timestamp(df_vars, df_protons)

df_final = utils.join_dataframes_on_timestamp(df_merged, df_vars)
df_final = df_final.sort_index()

if df_final.index.has_duplicates:
    print("Warning! Final dataframe has duplicate values of the index")

# Calculating analytical derived variables
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

stats = df_final.describe()

print(df_final.info())

df_final = df_final.reset_index() # So that Timestamp is a normal column in the CSV
df_final.to_csv("data/processed/wind_database.csv", index=False)
stats.to_csv("data/processed/wind_summary_stats.csv")

print("\nChecking for missing data:")
print(df_final.isna().sum()/len(df_final))
print("\n")
print("##############################################")