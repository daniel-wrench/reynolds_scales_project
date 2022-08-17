from utils import *
import glob

project_path = ''


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    project_path + 'data\\raw\\wi_elm2_3dp\\')]

df = pd.concat([pd.concat([pipeline(cdf_file_name, varlist=['Epoch', 'DENSITY', 'AVGTEMP', 'VELOCITY'],
                                    cadence='6H') for cdf_file_name in sub]) for sub in file_paths]).sort_index()

df.to_pickle(project_path + 'data\\processed\\wi_elm2_3dp_6hr.pkl')

print("Processed electron data")