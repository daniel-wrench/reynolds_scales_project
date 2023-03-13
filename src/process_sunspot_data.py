# Code to produce usable/cleaned csv file of sunspots
# Metadata at https://www.sidc.be/silso/infosndtot

import pandas as pd

sndf = pd.read_table("data/raw/sunspots/SN_d_tot_V2.0.txt", header=None, index_col=None)

df = pd.read_fwf(
    "data/raw/sunspots/SN_d_tot_V2.0.txt", 
    colspecs=[(0,4), (5,7), (8,10), (11,19), (21,24), (25,30), (32,35), (36,37)],
    names=['year', 'month', 'day', 'decimal_date', 'SN', 'SN_std', 'nb_obs', 'provisional']
)

df['Timestamp'] = pd.to_datetime(df[['year', 'month', 'day']])
df.drop(['year', 'month', 'day'], axis=1, inplace=True)

df.to_csv('data/processed/sunspot_dataset.csv', index=False)
print("\nFinished processing sunspot data\n")
