"""
get_raw_vars.py

Description:
    Processes raw CDF files to extract specified variables based on parameters defined in the `params` module.
    The processed data is then saved as a pickle file in the specified output directory.

Usage:
    python get_raw_vars.py [data_file_path] [variable_list] [thresholds] [cadence] [second_resolution]

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


"""

import datetime
import glob
import sys
import os

# Custom modules
import params
from utils import *

####### PARALLEL STUFF #######
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # Number of cores
rank = comm.Get_rank()  # Current core
status = MPI.Status()
##############################

# The following values should match the variables names in params.py 
# Vector components do not need to be specified, just the overall vector
sys_arg_dict = {
    # arg1
    "mag_path": params.mag_path,
    "omni_path": params.omni_path,
    "proton_path": params.proton_path,
    "electron_path": params.electron_path,

    # arg2
    "mag_vars": [params.timestamp, params.Bwind, params.Bwind_vec],
    "omni_vars": [params.timestamp, params.vsw, params.p, params.Bomni],
    "proton_vars": [params.timestamp, params.np, params.nalpha, params.Tp, params.Talpha, params.V_vec],
    "electron_vars": [params.timestamp, params.ne, params.Te],

    # arg3
    "mag_thresh": params.mag_thresh,
    "omni_thresh": params.omni_thresh,
    "proton_thresh": params.proton_thresh,
    "electron_thresh": params.electron_thresh,

    # arg4
    "dt_hr": params.dt_hr,
    "dt_protons": params.dt_protons,
    "int_size": params.int_size,

    # arg5
    "dt_lr": params.dt_lr
}

input_dir = "data/raw/" + sys_arg_dict[sys.argv[1]]
output_dir = "data/processed/" + sys_arg_dict[sys.argv[1]]

# input directory will already have been created by download data script
# output directory may still need to be created

if rank == 0:
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  else:
      # if it does exist, remove any existing files there: do not want confusion if
      # using a different number of cores
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

####### PARALLEL STUFF #######

# (Optionally) reducing the number of files for testing
# NEED TO COMMENT OUT OMNI FROM BASH FILE IF STARTING PART-WAY THROUGH
file_list = file_list[:5000]

list_of_lists = np.array_split(file_list, comm.size)

if len(list_of_lists) != comm.size:
    print("Number of lists does not equal number of cores!")

my_list = list_of_lists[rank]

##############################

dataframes = []

for file in my_list:
    try:
        temp_df = pipeline(
            file,
            varlist=sys_arg_dict[sys.argv[2]],
            thresholds=sys_arg_dict[sys.argv[3]],
            cadence=sys_arg_dict[sys.argv[4]]
        )
        print("Core {0:03d} reading {1}: {2:.2f}% missing".format(
            rank, file, temp_df.iloc[:, 0].isna().sum()/len(temp_df)*100))
        dataframes.append(temp_df)

    except Exception as e:
        print(f"Error reading {file}. Error: {e}; moving to next file")

df = pd.concat(dataframes)

# Ensuring observations are in chronological order
df = df.sort_index()
# NB: Using .asfreq() creates NA values

# Outputting pickle file
df.to_pickle(output_dir + sys_arg_dict[sys.argv[4]] + "_{:03d}.pkl".format(rank))

# Also outputting pickle at second resolution, if specified
if sys.argv[5] != "None":
    df = df.resample(sys_arg_dict[sys.argv[5]]).mean()
    df.to_pickle(output_dir + sys_arg_dict[sys.argv[5]] + "_{:03d}.pkl".format(rank))

    second_cadence = " and " + sys_arg_dict[sys.argv[5]]
else:
    second_cadence = ""

####### PARALLEL STUFF #######
# wait until all parallel processes are finished
comm.Barrier()
##############################

# Log statements
if rank == 0:
    print("\nProcessed {} files of {} data at {} cadence using {} cores\n".format(
        len(file_list),
        sys_arg_dict[sys.argv[1]],
        sys_arg_dict[sys.argv[4]] + second_cadence,
        comm.size))
    print("##################################\n")
    print(datetime.datetime.now())
