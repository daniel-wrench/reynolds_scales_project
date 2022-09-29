from utils import *
import glob

project_path = ''


def get_subfolders(path):
    return glob.glob(path + '/*')


def get_cdf_paths(subfolder):
    return glob.iglob(subfolder + '/*.cdf')


file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(
    project_path + 'data/raw/wind/3dp/3dp_plsp/')]

# View raw CDF info

# cdf = read_cdf(next(file_paths[0]))
# pprint(cdf.cdf_info())

# #cdf.varattsget(variable='MOM.P.DENSITY', expand=True)
# cdf.varget("MOM.P.DENSITY")

# YOU MUST RE-RUN file_paths DEFINITION BEFORE THE FOLLOWING IF USING ABOVE LINES

df = pd.concat([
    pd.concat([
        pipeline(
            cdf_file_name,
            varlist=['Epoch', 'MOM.P.DENSITY',
                     'MOM.P.AVGTEMP'],
            cadence='12H')
        for cdf_file_name in sub])
    for sub in file_paths]).sort_index()

df.to_pickle(project_path + 'data/processed/wi_plsp_3dp_12hr.pkl')

print("\n\nProcessed proton data:\n")
print(df.head())
print(datetime.datetime.now())

