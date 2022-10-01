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
    "dt_lr": params.dt_lr,
    "dt_hr": params.dt_hr,
    "int_size": params.int_size
}

input_dir = 'data/raw/' + sys_arg_dict[sys.argv[1]]
output_dir = 'data/processed/' + sys_arg_dict[sys.argv[1]]

# input directory will already have been created by download data script
# output directory may still need to be created
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    input_dir)]

# View raw CDF info

# cdf = read_cdf(next(file_paths[0]))
# pprint(cdf.cdf_info())

# cdf.varattsget(variable='F', expand=True)
# cdf.varget("F")

df = pd.DataFrame({})

for sub in file_paths:
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

df.to_pickle(output_dir + sys_arg_dict[sys.argv[4]] + '.pkl')

print("\nProcessed {} data at {} cadence:\n".format(
    sys_arg_dict[sys.argv[1]], sys_arg_dict[sys.argv[4]]))
print(df.head())
print("\n")
print(datetime.datetime.now())
