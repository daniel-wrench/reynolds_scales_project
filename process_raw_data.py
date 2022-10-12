from utils import *
import datetime
import glob
import math
import params
import sys
import os
from mpi4py import MPI 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

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

input_dir = 'data/raw/' + params.omni_path
output_dir = 'data/processed/' + params.omni_path

# input directory will already have been created by download data script
# output directory may still need to be created
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_subfolders(path):
    return sorted(glob.glob(path + '/*'))

def get_cdf_paths(subfolder):
    return sorted(glob.iglob(subfolder + '/*.cdf'))

file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    input_dir)]

file_list = []
for sub in file_paths:
    for cdf_file_name in sub:
        file_list.append(cdf_file_name)

# View raw CDF info

# cdf = read_cdf(file_paths[0][0])
# pprint(cdf.cdf_info())

# pprint(cdf.varattsget(variable='BGSE', expand=True))
# cdf.varget("Epoch")

n = comm.size

# Getting list of lists of files for each core

def getSublists(lst,n):
    subListLength = math.ceil(len(lst)/n)
    for i in range(0, len(lst), subListLength):
        yield lst[i:i+subListLength]

list_of_lists = list(getSublists(file_list,n))

if len(list_of_lists) != n:
    print("Number of lists does not equal n!")

my_list=list_of_lists[rank]
# Iterating over each list, turning each CDF to a dataframe

df = pd.DataFrame({})

# A generator object might be faster here
for file in my_list:
    print("Reading " + file)
    try:
        temp_df = pipeline(
            file,
            varlist=[params.timestamp, params.vsw, params.p, params.Bomni],
            thresholds=params.omni_thresh,
            cadence=params.int_size
        )
        df = pd.concat([df, temp_df])
    except:
        print("Error reading CDF file; moving to next file")
        nan_df = pd.DataFrame({})  # empty dataframe
        df = pd.concat([df, nan_df])
    
    # Ensuring observations are in chronological order
    df = df.sort_index() 
    # NB: Using .asfreq() creates NA values

    # Checking for missing data
    if df.isna().any().sum() != 0:
        print("MISSING DATA ALERT!")
        print(df.isna().sum()/len(df))
    
    df.to_pickle(output_dir + params.int_size + "_{:03d}.pkl".format(rank))

        # Also outputting pickle at second resolution, if specified
        # if sys.argv[5] !="None":
        #     df = df.resample(sys_arg_dict[sys.argv[5]]).mean()
        #     df.to_pickle(output_dir + sys_arg_dict[sys.argv[5]] + '.pkl')

        #     print("\nProcessed {} data at {} cadence:\n".format(
        #     params.omni_path, sys_arg_dict[sys.argv[5]]))
        #     print(df.info())
        #     print(df.head())
        #     print("\nChecking for missing data:")
        #     print(df.isna().sum()/len(df))
        #     print("##################################\n")
        #     print(datetime.datetime.now())

## Alt method

# subListLength = math.ceil(len(file_list)/n)
# for i in range(0, len(file_list), subListLength):
#     file_list_subset = file_list[i:i+subListLength]

#     df = pd.DataFrame({})

#     # A generator object might be faster here
#     for file in file_list_subset:
#         print("Reading " + file)
#         try:
#             temp_df = pipeline(
#                 file,
#                 varlist=[params.timestamp, params.vsw, params.p, params.Bomni],
#                 thresholds=params.omni_thresh,
#                 cadence=params.int_size
#             )
#             df = pd.concat([df, temp_df])
#         except:
#             print("Error reading CDF file; moving to next file")
#             nan_df = pd.DataFrame({})  # empty dataframe
#             df = pd.concat([df, nan_df])
#         df.to_pickle(output_dir + params.int_size + "_" + str(i) + '.pkl')

print(sys.argv[1])
print("\nProcessed {} data at {} cadence\n".format(
    params.omni_path, params.int_size))
print("##################################\n")
print(datetime.datetime.now())
