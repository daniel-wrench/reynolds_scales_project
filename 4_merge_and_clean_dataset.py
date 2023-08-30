
# TEMP CODE WHEN NEEDING TO MERGE FILES (i.e., did not run on all data at once)

import numpy as np
import pandas as pd
import utils

df_1 = pd.read_csv("data/processed/wind_database_1995_1998.csv")
df_2 = pd.read_csv("data/processed/wind_database_1999_2007.csv")
df_3 = pd.read_csv("data/processed/wind_database_2007_2022.csv")

df_1 = df_1.set_index("Timestamp").sort_index()
df_2 = df_2.set_index("Timestamp").sort_index()
df_3 = df_3.set_index("Timestamp").sort_index()
df_omni = df_1[["vsw", "p", "Bomni"]]

# We have the entire OMNI data in each dataframe
# We need to exclude it so it doesn't get added together during the following merging process
# which takes into account the ragged transitions from one df to the next

df_merged = pd.concat([df_1, df_2, df_3], verify_integrity=False)
df_merged = df_merged.drop(["vsw", "p", "Bomni"], axis=1)
df_merged.index.has_duplicates

# # Can also check for duplicate timestamps during the concatentation with the following: 
# #df_merged = pd.concat([df_1, df_2], verify_integrity=True)
# #ValueError: Indexes have overlapping values

df_merged = df_merged.groupby(df_merged.index).agg(sum)
# Dealing with any resultant 0s from summing to NAs together
df_merged = df_merged.replace(0, np.nan)

df_merged.index = pd.to_datetime(df_merged.index)
df_omni.index = pd.to_datetime(df_omni.index)
df = utils.join_dataframes_on_timestamp(df_merged, df_omni)
df.index.has_duplicates

# # Checking merge (border between end of first file and start of second, with a ragged transition)
# # df_merged_final["1998-12-30":"1999-01-03"]

df.rename(columns={"tb":"fb"}, inplace=True)

# df[["tcf", "ttc", "Re_di", "Re_lt", "Re_lt_u", "Re_tb"]].describe()
# np.mean(df.Re_lt).round(-4)
# np.mean(df.Re_di).round(-4)
# np.mean(df.Re_tb).round(-4)

# df[["di", "vsw", "ttk", "ttu", "ttc", "Re_di", "Re_lt", "Re_tb"]].describe().round(2)
# # CHECK MAX VALS

df.to_csv("data/processed/wind_database.csv")

#####################################################

df = pd.read_csv("data/processed/wind_omni_dataset.csv")
df.Timestamp = pd.to_datetime(df.Timestamp)
df.set_index("Timestamp", inplace=True)
df.sort_index(inplace=True)

#### DATA CLEANING (subsetting and dealing with outliers)

df_l1 = df["2004-06-01":]

# Few timestamps (0.1%) have ttc < 0 
# All of these have unusually large values for qk. Total of 5% have qk > -1.7
# 3 values also have ttu < 1

# Here I am removing all these values
# Removing NAs as well, this brings my total number of rows down to about 18,000
# It only slightly decreases the qk mean from -2.63 to -2.69, but it does
# remove very large Re_lt outliers, reducing the mean from 4,500,000 to 160,000
# It still leaves around 2% of rows where qk > qi

#Counting outliers

df_l1.loc[:, "small_ttu"] = 0 
df_l1.loc[:, "qk > -1.7"] = 0
df_l1.loc[:, "qk > qi"] = 0

df_l1.loc[df_l1["ttu"] < 1, "small_ttu"] = 1
df_l1.loc[df_l1["qk"] > -1.7, "qk > -1.7"] = 1
df_l1.loc[df_l1["qk"] > df_l1["qi"], "qk > qi"] = 1

df_l1[["small_ttu", "qk > -1.7", "qk > qi"]].mean()
df_l1.groupby(["qk > -1.7", "qk > qi", "small_ttu"])[["small_ttu", "qk > -1.7", "qk > qi"]].value_counts()
df_l1.drop(["small_ttu", "qk > -1.7", "qk > qi"], axis=1, inplace=True)

# Removing outlier slope rows
df_l1_cleaned = df_l1[df_l1.qk < -1.7]
df_l1_cleaned = df_l1_cleaned[df_l1_cleaned.ttu > 1] # not needed for L1 range

# Removing negative tci values (only 5, numerical issue with finding argmin)
df_l1_cleaned.loc[df_l1_cleaned.tci < 0, ["tci", "lambda_c_int"]] = np.nan

df_l1_cleaned.to_csv("data/processed/wind_dataset_l1_cleaned.csv", index=True)

corr_table = df_l1_cleaned.corr()
corr_table.to_csv("wind_dataset_l1_cleaned_corr.csv")

key_vars = df_l1_cleaned[["lambda_c_fit", "lambda_c_int", "lambda_c_e", "lambda_t_raw", "qi", "qk", "fb", "lambda_t", "Re_lt", "Re_di", "Re_tb"]]

key_stats = key_vars.describe().round(2)
key_stats.to_csv("wind_dataset_l1_cleaned_key_stats.csv")