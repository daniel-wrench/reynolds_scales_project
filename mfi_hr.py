from utils import *
import glob

project_path = ''

def get_subfolders(path):
  return glob.glob(path + '/*')

def get_cdf_paths(subfolder):
  return glob.iglob(subfolder + '/*.cdf')

taylor_paths = [[get_cdf_paths(subfolder) for subfolder in get_subfolders(project_path + 'full_data\\WI_H2_MFI')][3]] 
# Index = 3 corresponds to 2019
# Changed / in file path to \\
print(list(taylor_paths))

wih2mfi_hr = pd.concat([pd.concat([pipeline(cdf_file_name, varlist=['Epoch', 'BF1', 'BGSE'],
                    cadence=convert_cadence_to_string(0.1)) for cdf_file_name in sub]) for sub in taylor_paths]).sort_index()

# for sub in taylor_paths:
#   for cdf_file_name in sub:
#     print(cdf_file_name)
#     df = convert_cdf_to_dataframe(read_cdf(cdf_file_name), varlist=['Epoch', 'BF1', 'BGSE'])
#     print(df.head())

wih2mfi_hr = wih2mfi_hr.sort_index()

wih2mfi_hr.to_pickle(project_path + 'wih2mfi_hr_19')
