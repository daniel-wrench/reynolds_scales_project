import pandas as pd
import glob
import os

file_paths = sorted(glob.iglob("data/processed/dataset_*.pkl"))

print("MERGING DATASETS")
df_merged = pd.DataFrame({})
for file in file_paths:
    print("Reading " + file)
    df_merged = pd.concat([df_merged, pd.read_pickle(file)])
    os.remove(file)
df_merged = df_merged.sort_index()

stats = df_merged.describe()

print("\nSAVING MERGED DATASET AND SUMMARY STATS TABLE\n")
print(df_merged.info())

df_merged.to_csv("data/processed/db_wind.csv", index=False)
stats.to_csv("data/processed/db_wind_summary_stats.csv")

print("\nChecking for missing data:")
print(df_merged.isna().sum()/len(df_merged))
print("\n")
print("##############################################")
