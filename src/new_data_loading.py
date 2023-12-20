
import datetime
import glob
import numpy as np
from src.utils import *
import src.params as params

start_date = "20160101"
end_date = "20160107"
num_cores = 2
rank = 0


def get_subfolders(path):
    return sorted(glob.glob(path + "/*"))


def get_cdf_paths(subfolder):
    return sorted(glob.iglob(subfolder + "/*.cdf"))


def get_file_list(input_dir):
    file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(input_dir)]
    file_list = []
    for sub in file_paths:
        for cdf_file_name in sub:
            file_list.append(cdf_file_name)        
    return file_list


def generate_date_strings(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    date_list = [start + datetime.timedelta(days=x) for x in range((end-start).days + 1)]
    return [date.strftime("%Y%m%d") for date in date_list]


def read_dated_file(date, file_list, varlist, newvarnames, cadence, thresholds):
    matched_files = [file for file in file_list if date in file]
    if not matched_files:
        print(f"No files found for date {date}")
    elif len(matched_files) > 1:
        print(f"Multiple files found for date {date}")
        print(matched_files)
    else:
        # Read in file
        try:
            df = pipeline(
                matched_files[0],
                varlist=varlist,
                thresholds=thresholds,
                cadence=cadence
            )
            print("\nCore {0:03d} reading {1}: {2:.2f}% missing".format(
                rank, matched_files[0], df.iloc[:, 0].isna().sum()/len(df)*100))
            df = df.rename(columns=newvarnames)
            print(df.head())
            return df
            #pd.to_pickle(df_mfi, "data/processed/wind/mfi/mfi_h2/" + date + ".pkl")

        except Exception as e:
            print(f"\nError reading {matched_files[0]}. Error: {e}; moving to next file")
            pass


mfi_file_list = get_file_list("data/raw/wind/mfi/mfi_h2/")
proton_file_list = get_file_list("data/raw/wind/3dp/3dp_pm/")
electron_file_list = get_file_list("data/raw/wind/3dp/3dp_elm2/")

# Generate all date strings
all_dates = generate_date_strings(start_date, end_date)

# Split date strings among cores
dates_for_cores = np.array_split(all_dates, num_cores)


# For each core, read in files for each date
for date in dates_for_cores[rank]:

    # # MFI
    mfi_df_hr = read_dated_file(date, 
                    mfi_file_list, 
                    [params.timestamp, params.Bwind, params.Bwind_vec], 
                    {params.Bx: "Bx", params.By: "By", params.Bz: "Bz"},
                    params.dt_hr, 
                    params.mag_thresh
                    )
    
    # Getting low-res version for correlation fn calculation later on
    mfi_df_lr = mfi_df_hr.resample(params.dt_lr).mean()

    # PROTONS
    proton_df = read_dated_file(date,
                    proton_file_list,
                    [params.timestamp, params.np, params.nalpha, params.Tp, params.Talpha, params.V_vec],
                    {params.Vx: "Vx",
                     params.Vy: "Vy",
                     params.Vz: "Vz",
                     params.np: "np",
                     params.nalpha: "nalpha",
                     params.Tp: "Tp",
                     params.Talpha: "Talpha"},
                     params.dt_protons,
                     params.proton_thresh
                     )
    
    electron_df = read_dated_file(date,
                    electron_file_list,
                    [params.timestamp, params.ne, params.Te],
                    {params.ne: "ne", params.Te: "Te"},
                    params.int_size,
                    params.electron_thresh
                    )
