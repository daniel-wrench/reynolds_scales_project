from utils import *
import glob

project_path = ''


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    project_path + 'data/raw/wind/mfi/mfi_h2/')]

# df = pd.concat([pd.concat([pipeline(cdf_file_name, varlist=['Epoch', 'BF1', 'BGSE'],
#                                     cadence='5S') for cdf_file_name in sub]) for sub in file_paths]).sort_index()

df = pd.DataFrame({})

for sub in file_paths:
    for cdf_file_name in sub:
        print("Reading " + cdf_file_name) 
        try:
            temp_df = pipeline(
                cdf_file_name, 
                varlist=['Epoch', 'BF1', 'BGSE'], 
                cadence='5S'
            )
            df = pd.concat([df, temp_df])
        except:
            print("Error reading CDF file; moving to next file")
            nan_df = pd.DataFrame({}) # empty dataframe
            df = pd.concat([df, nan_df])

#df.to_pickle(project_path + 'data/processed/wi_h2_mfi_lr.pkl')

print("\n\nProcessed Wind MFI low-res data:\n")
print(df.head())
print(datetime.now())
