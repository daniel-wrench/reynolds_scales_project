import pandas as pd
import glob
import os
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

# Bringing in omni data. This needs to be brought in separately as its files are monthly in size, rather than daily
# Also, we do not calculate any secondary variables from the OMNI variables, so we do not need to do this in construct_database.py

omni_file_paths = sorted(glob.iglob("data/processed/" + params.omni_path + params.int_size + "_*.pkl"))

print("MERGING OMNI DATASETS")

df_omni = pd.DataFrame({})
for omni_file in omni_file_paths:
    print("Reading " + omni_file)
    df_omni = pd.concat([df_omni, pd.read_pickle(omni_file)])
    #os.remove(omni_file)

df_omni = df_omni.rename(
    columns={
        params.vsw: 'vsw',
        params.p: 'p',
        params.Bomni: 'Bomni'})


print("\nSAVING FULL MERGED DATASET AND SUMMARY STATS TABLE\n")

df_final = utils.join_dataframes_on_timestamp(df_merged, df_omni)
stats = df_final.describe()

print(df_final.info())

df_final = df_final.reset_index() # So that Timestamp is a normal column in the CSV
df_final.to_csv("data/processed/db_wind.csv", index=False)
stats.to_csv("data/processed/db_wind_summary_stats.csv")

print("\nChecking for missing data:")
print(df_final.isna().sum()/len(df_final))
print("\n")
print("##############################################")
