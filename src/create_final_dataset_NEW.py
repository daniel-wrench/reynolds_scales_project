
import os
import pandas as pd
import utils

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

stats = merged_dataframe.describe()
print(merged_dataframe.info())
merged_dataframe.head()

# Output the merged dataframe as a CSV file
output_csv_path = 'wind_omni_dataset_WEEK_NEW.csv'  # Specify your output path
merged_dataframe = merged_dataframe.reset_index() # So that Timestamp is a normal (first) column in the CSV
merged_dataframe.to_csv(output_csv_path, index=False)

print(f'Merged DataFrame saved as CSV at: {output_csv_path}')
