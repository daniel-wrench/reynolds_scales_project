
import os
import pandas as pd
import numpy as np
import src.utils as utils

# Specify the folder path containing the pickle files
folder_path = 'data/processed'  # Replace with your folder path

# List all pickle files in the folder
pickle_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

# Import and concatenate all the dataframes
dataframes = [pd.read_pickle(os.path.join(folder_path, file)) for file in pickle_files]
merged_dataframe = pd.concat(dataframes)

# Bringing in sunspot data
df_ss = pd.read_csv("data/processed/sunspot_dataset.csv")
df_ss["Timestamp"] = pd.to_datetime(df_ss["Timestamp"])
df_ss.set_index("Timestamp", inplace=True)
# Limit to only the sunspot number column
df_ss = df_ss['SN']
# Limit to only the range of other data
df_ss = df_ss[merged_dataframe.index.min():merged_dataframe.index.max()+pd.Timedelta("12H")]
df_ss = df_ss.resample("12H").agg("ffill")[:-1] # Up-sampling to twice daily; removing surplus final row

merged_dataframe = utils.join_dataframes_on_timestamp(merged_dataframe, df_ss)

# Rearrange columns (based on units)
merged_dataframe = merged_dataframe[[
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
    'dv', 
    'va', 
    'db_a', 
    'vte', 
    'vtp', 
    'zp', 
    'zm', 
    'zp_decay', 
    'zm_decay', 
    'p' 
    ]]

merged_dataframe = merged_dataframe.sort_index()
print(merged_dataframe.info())
print(merged_dataframe.head())

# Output the merged dataframe as a CSV file
output_csv_path = 'wind_dataset.csv'
merged_dataframe.to_csv(output_csv_path, index=True)
print(f'Merged DataFrame saved as CSV at: {output_csv_path}')

stats = merged_dataframe.describe().round(2)
stats.to_csv("wind_dataset_stats.csv")

corr_matrix = merged_dataframe.corr()
corr_matrix.to_csv("wind_dataset_l1_cleaned_corr_matrix.csv")


#####################################################

# SUBSETTING TO L1 AND CLEANING OUTLIERS

# df = pd.read_csv("wind_dataset.csv")
# df.Timestamp = pd.to_datetime(df.Timestamp)
# df.set_index("Timestamp", inplace=True)
# df.sort_index(inplace=True)

#### DATA CLEANING (subsetting and dealing with outliers)

df_l1 = merged_dataframe["2004-06-01":] 
# Wind has been at L1 continuously since June 2004: https://wind.nasa.gov/orbit.php 

# Few timestamps (0.1%) have ttc < 0 
# All of these have unusually large values for qk. Total of 5% have qk > -1.7
# 3 values also have ttu < 1

# Here I am removing all these values
# Removing NAs as well, this brings my total number of rows down to about 18,000
# It only slightly decreases the qk mean from -2.63 to -2.69, but it does
# remove very large Re_lt outliers, reducing the mean from 4,500,000 to 160,000
# It still leaves around 2% of rows where qk > qi

# Counting outliers using outlier flag columns
df_l1.loc[:, "small_ttu"] = 0 
df_l1.loc[:, "qk > -1.7"] = 0
df_l1.loc[:, "qk > qi"] = 0

df_l1.loc[df_l1["ttu"] < 1, "small_ttu"] = 1
df_l1.loc[df_l1["qk"] > -1.7, "qk > -1.7"] = 1
df_l1.loc[df_l1["qk"] > df_l1["qi"], "qk > qi"] = 1

df_l1[["small_ttu", "qk > -1.7", "qk > qi"]].mean()
df_l1.groupby(["qk > -1.7", "qk > qi", "small_ttu"])[["small_ttu", "qk > -1.7", "qk > qi"]].value_counts()

df_l1.drop(["small_ttu", "qk > -1.7", "qk > qi"], axis=1, inplace=True)

# Removing outlier slope rows
df_l1_cleaned = df_l1[df_l1.qk < -1.7]
# df_l1_cleaned = df_l1_cleaned[df_l1_cleaned.ttu > 1] # not needed for L1 range

# Removing negative tci values (only 6, numerical issue with finding argmin)
df_l1_cleaned.loc[df_l1_cleaned.tci < 0, ["tci", "lambda_c_int"]] = np.nan

# Saving cleaned dataset
df_l1_cleaned.to_csv("wind_dataset_l1_cleaned.csv", index=True)

# Saving correlations and summary statistics
corr_table = df_l1_cleaned.corr()
corr_table.to_csv("wind_dataset_l1_cleaned_corr.csv")

stats = df_l1_cleaned.describe().round(2)
stats.to_csv("wind_dataset_l1_cleaned_stats.csv")