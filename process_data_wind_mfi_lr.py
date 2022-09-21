from utils import *
import glob

project_path = ''


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    project_path + 'data\\raw\\wind\\mfi\\mfi_h2\\')]

df = pd.concat([pd.concat([pipeline(cdf_file_name, varlist=['Epoch', 'BF1', 'BGSE'],
                                    cadence='5S') for cdf_file_name in sub]) for sub in file_paths]).sort_index()

df.to_pickle(project_path + 'data\\processed\\wi_h2_mfi_lr.pkl')

print("Processed Wind MFI low-res data")