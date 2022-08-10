from utils import *
import glob

project_path = ''

def get_subfolders(path):
  return glob.glob(path + '/*')

def get_cdf_paths(subfolder):
  return glob.iglob(subfolder + '/*.cdf')

corr_paths = [[get_cdf_paths(subfolder) for subfolder in get_subfolders(project_path + 'data\\raw\\wi_h2_mfi')][0]]
# Index = 0 corresponds to 2016

wih2mfi_lr = pd.concat([pd.concat([pipeline(cdf_file_name, varlist=['Epoch', 'BF1', 'BGSE'],
                    cadence='5S') for cdf_file_name in sub]) for sub in corr_paths]).sort_index()

wih2mfi_lr = wih2mfi_lr.sort_index()

wih2mfi_lr.to_pickle(project_path + 'data\\processed\\wi_h2_mfi_lr_16')
