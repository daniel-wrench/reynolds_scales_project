# Reads in Wind MFI CDF files
# Extracts desired variables
# Re-samples to desired cadence
# Saves to .pkl file

from utils import *
import glob

project_path = ''


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    project_path + 'data\\raw\\wi_h2_mfi\\')]

df = pd.concat([
        pd.concat([
            pipeline(
                cdf_file_name, 
                varlist=['Epoch', 'BF1', 'BGSE'], 
                cadence='0.091S'
            ) 
            for cdf_file_name in sub
        ]) 
        for sub in file_paths
    ]).sort_index()

df.to_pickle(project_path + 'data\\processed\\wi_h2_mfi_hr.pkl')

print("Processed Wind MFI high-res data")