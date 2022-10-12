from utils import *
import sys
import datetime
import glob
import params
import os

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

input_dir = 'data/raw/' + sys_arg_dict[sys.argv[1]]
output_dir = 'data/processed/' + sys_arg_dict[sys.argv[1]]

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

n = 3

#Getting list of lists of files for each core

def getSublists(lst,n):
    subListLength = math.ceil(len(lst)/n)
    for i in range(0, len(lst), subListLength):
        yield lst[i:i+subListLength]

list_of_lists = list(getSublists(file_list,n))

if len(list_of_lists) != n:
    print("Number of lists does not equal n!")

for i in range(n):
df = pd.DataFrame({})

# A generator object might be faster here
    for file in list_of_lists[i]:
    print("Reading " + file)
        try:
            temp_df = pipeline(
            file,
                varlist=sys_arg_dict[sys.argv[2]],
                thresholds=sys_arg_dict[sys.argv[3]],
                cadence=sys_arg_dict[sys.argv[4]]
            )
            df = pd.concat([df, temp_df])
        except:
            print("Error reading CDF file; moving to next file")
            nan_df = pd.DataFrame({})  # empty dataframe
            df = pd.concat([df, nan_df])

# Ensuring observations are in chronological order
df = df.sort_index() 
# NB: Using .asfreq() creates NA values

df.to_pickle(output_dir + sys_arg_dict[sys.argv[4]] + '.pkl')

print("\nProcessed {} data at {} cadence:\n".format(
    sys_arg_dict[sys.argv[1]], sys_arg_dict[sys.argv[4]]))
print(df.info())
print(df.head())
print("\nChecking for missing data:")
print(df.isna().sum()/len(df))
print("##################################\n")
print(datetime.datetime.now())

# Also outputting pickle at second resolution, if specified
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


print("\nProcessed {} data at {} cadence\n".format(
    params.omni_path, params.int_size))
    print("##################################\n")
    print(datetime.datetime.now())
