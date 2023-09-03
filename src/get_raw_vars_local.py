"""
get_raw_vars_local.py

Description:
    Processes raw CDF files to extract specified variables based on parameters defined in the `params` module.
    The processed data is then saved as a pickle file in the specified output directory.

Usage:
    python get_raw_vars_local.py [data_file_path] [variable_list] [thresholds] [cadence] [second_resolution]

Arguments:
    data_file_path: Specifies the location of the input data files (e.g., mag_path, omni_path, etc.).
    variable_list: List of variables to extract from the CDF files (e.g., mag_vars, omni_vars, etc.).
    thresholds: Threshold values for the data processing.
    cadence: The primary time resolution for the processed data.
    second_resolution: An optional second time resolution for the processed data. Use "None" if not required.

Outputs:
    Pickle files containing the processed data, saved in the `data/processed/` directory.

Note:
    Ensure the required modules and dependencies are installed and the necessary data files are available in the `data/raw/` directory.

Author:
    Daniel Wrench
Date:
    3/9/2023

"""

import datetime
import glob
import params
import sys
import os

from utils import *

sys_arg_dict = {
    # arg1
    "mag_path": params.mag_path,
    "omni_path": params.omni_path,
    "proton_path": params.proton_path,
    "electron_path": params.electron_path,

    # arg2
    "mag_vars": [params.timestamp, params.Bwind, params.Bwind_vec],
    "omni_vars": [params.timestamp, params.vsw, params.p, params.Bomni],
    "proton_vars": [params.timestamp, params.ni, params.Ti],
    "electron_vars": [params.timestamp, params.ne, params.Te],

    # arg3
    "mag_thresh": params.mag_thresh,
    "omni_thresh": params.omni_thresh,
    "proton_thresh": params.proton_thresh,
    "electron_thresh": params.electron_thresh,

    # arg4
    "dt_hr": params.dt_hr,
    "int_size": params.int_size,

    # arg5
    "dt_lr": params.dt_lr
}

input_dir = "data/raw/" + sys_arg_dict[sys.argv[1]]
output_dir = "data/processed/" + sys_arg_dict[sys.argv[1]]

# input directory will already have been created by download data script
# output directory may still need to be created
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    # if it does exist, remove any existing files there: do not want confusion
    for file in glob.glob(output_dir + "*"):
        os.remove(file)


def get_subfolders(path):
    return sorted(glob.glob(path + "/*"))


def get_cdf_paths(subfolder):
    return sorted(glob.iglob(subfolder + "/*.cdf"))


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(input_dir)]

file_list = []
for sub in file_paths:
    for cdf_file_name in sub:
        file_list.append(cdf_file_name)

# View raw CDF info

# cdf = read_cdf(file_paths[0][0])
# pprint(cdf.cdf_info())

# pprint(cdf.varattsget(variable="BGSE", expand=True))
# cdf.varget("Epoch")

dataframes = []

for file in file_list:
    try:
        temp_df = pipeline(
            file,
            varlist=sys_arg_dict[sys.argv[2]],
            thresholds=sys_arg_dict[sys.argv[3]],
            cadence=sys_arg_dict[sys.argv[4]]
        )
        print("Reading {0}: {1:.2f}% missing".format(file, temp_df.iloc[:, 0].isna().sum()/len(temp_df)*100))
        dataframes.append(temp_df)

    except Exception as e:
        print(f"Error reading {file}. Error: {e}; moving to next file")

df = pd.concat(dataframes)

# Ensuring observations are in chronological order
df = df.sort_index()
# NB: Using .asfreq() creates NA values

# Outputting pickle file
df.to_pickle(output_dir + sys_arg_dict[sys.argv[4]] + ".pkl")

# Also outputting pickle at second resolution, if specified
if sys.argv[5] != "None":
    df = df.resample(sys_arg_dict[sys.argv[5]]).mean()
    df.to_pickle(output_dir + sys_arg_dict[sys.argv[5]] + ".pkl")

    second_cadence = " and " + sys_arg_dict[sys.argv[5]]
else:
    second_cadence = ""

# Log statements
print("\nProcessed {} files of {} data at {} cadence\n".format(
    len(file_list),
    sys_arg_dict[sys.argv[1]],
    sys_arg_dict[sys.argv[4]] + second_cadence))

print("##################################\n")
print(datetime.datetime.now())
