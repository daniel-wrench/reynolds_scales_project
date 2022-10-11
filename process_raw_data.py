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

# View raw CDF info

# cdf = read_cdf(file_paths[0][0])
# pprint(cdf.cdf_info())

# pprint(cdf.varattsget(variable='BGSE', expand=True))
# cdf.varget("Epoch")

df = pd.DataFrame({})

for sub in file_paths:
    # If you want to test on only n files in the directory, change the below to 
    #for cdf_file_name in list(sub)[:3]:
    # (otherwise I think this generator object is faster)
    for cdf_file_name in sub:
        print("Reading " + cdf_file_name)
        try:
            temp_df = pipeline(
                cdf_file_name,
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
if sys.argv[5] !="None":
    df =df.resample(sys_arg_dict[sys.argv[5]]).mean()
    df.to_pickle(output_dir + sys_arg_dict[sys.argv[5]] + '.pkl')

    print("\nProcessed {} data at {} cadence:\n".format(
    sys_arg_dict[sys.argv[1]], sys_arg_dict[sys.argv[5]]))
    print(df.info())
    print(df.head())
    print("\nChecking for missing data:")
    print(df.isna().sum()/len(df))
    print("##################################\n")
    print(datetime.datetime.now())
