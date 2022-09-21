from utils import *
import glob

project_path = ''


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')

file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    project_path + 'data/raw/omni/omni_cdaweb/hro2_1min/')] 
# Previously had double backslashes in filepath, but did not work in Raapoi

## For debugging

# for sub in file_paths:
#     for cdf_file_name in sub:
#         print(cdf_file_name)

## View raw CDF info

# cdf = read_cdf(next(file_paths[0]))
# pprint(cdf.cdf_info())

# cdf.varattsget(variable='F', expand=True)
# cdf.varget("F")

# YOU MUST RE-RUN file_paths DEFINITION BEFORE THE FOLLOWING IF USING ABOVE LINES

df = pd.concat([
        pd.concat([
            pipeline(
                cdf_file_name,
                varlist = [
                    'Epoch',
                    'flow_speed',
                    'Pressure',
                    'F'
                ], 
                thresholds={
                    'flow_speed': [0, 1000],
                    'Pressure': [0, 200],
                    'F': [0,50]
                },
                cadence='12H'
            ) 
            for cdf_file_name in sub
        ]) 
        for sub in file_paths
    ]).sort_index()

df.to_pickle(project_path + 'data/processed/omni_12hr.pkl')

print("Processed OMNI data")
