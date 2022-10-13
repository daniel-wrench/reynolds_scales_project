import pandas as pd
import glob
import params
import os
import sys

# Smaller version of the one in process_raw_data.py
sys_arg_dict = {
    # arg1
    "mag_path": params.mag_path,
    "omni_path": params.omni_path,
    "proton_path": params.proton_path,
    "electron_path": params.electron_path,

    # arg2
    "int_size": params.int_size,
    "dt_hr": params.dt_hr,
    "dt_lr": params.dt_lr
}

dir_path = 'data/processed/' + sys_arg_dict[sys.argv[1]] + sys_arg_dict[sys.argv[2]]
file_paths = sorted(glob.iglob(dir_path + '_*.pkl'))

df_merged = pd.DataFrame({})
for file in file_paths:
    df_merged = pd.concat([df_merged, pd.read_pickle(file)])
    os.remove(file)
df_merged = df_merged.sort_index()

print("MERGED DATAFRAMES FROM "+ dir_path)
print("\nChecking for missing data:")
print(df_merged.isna().sum()/len(df_merged))
print("\n")
print(df_merged.info())
print("\n")
print(df_merged.head())
print("...")
print(df_merged.tail())
print("##############################################")

df_merged.to_pickle(dir_path + ".pkl")
#rm data/processed/wind/3dp/3dp_elm2/12H_*.pkl
