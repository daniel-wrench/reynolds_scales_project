
# TEMP CODE WHEN NEEDING TO MERGE FILES (i.e., did not run on all data at once)

# CONFIRM BELOW, THEN RE-RUN 3_CALCULATE_ANALYTICAL_VARS.PY ON RAAPOI,
# THEN RUN 5_PLOT_FIGURES.PY

import numpy as np
import pandas as pd

import src.utils as utils

df_1 = pd.read_csv("wind_omni_dataset_FIRST_HALF.csv")
df_2 = pd.read_csv("wind_omni_dataset_SECOND_HALF.csv")

df_1 = df_1.set_index("Timestamp").sort_index()
df_2 = df_2.set_index("Timestamp").sort_index()
df_omni = df_1[["Vomni", "pomni", "Bomni"]]

# We have the entire OMNI data in each dataframe
# We need to exclude it so it doesn't get added together during the following merging process
# which takes into account the ragged transitions from one df to the next

df_merged = pd.concat([df_1, df_2], verify_integrity=False)
df_merged = df_merged.drop(["Vomni", "pomni", "Bomni"], axis=1)
df_merged.index.has_duplicates # expecting True

# # Can also check for duplicate timestamps during the concatentation with the following: 
# #df_merged = pd.concat([df_1, df_2], verify_integrity=True)
# #ValueError: Indexes have overlapping values

df_merged = df_merged.groupby(df_merged.index).agg(sum)
# Dealing with any resultant 0s from summing to NAs together
df_merged = df_merged.replace(0, np.nan)

df_merged.index = pd.to_datetime(df_merged.index)
df_omni.index = pd.to_datetime(df_omni.index)
df = utils.join_dataframes_on_timestamp(df_merged, df_omni)
df.index.has_duplicates # expecting False

# # Checking merge (border between end of first file and start of second, with a ragged transition)
check = df_merged["1998-12-30":"1999-01-03"]

# Overall % missing
missing_percentage = df.isna().mean() * 100
print("Percentage of missing values per column:")
print(missing_percentage.sort_values(ascending=False))

# Quick analysis of % difference of values from the 3 omni variables

df["Vomni_diff"] = (df["V0"] - df["Vomni"]) / df["V0"]
df["pomni_diff"] = (df["p"] - df["pomni"]) / df["p"]
df["Bomni_diff"] = (df["B0"] - df["Bomni"]) / df["B0"]

df[["Vomni_diff", "pomni_diff", "Bomni_diff"]].describe().round(2)
# Typical difference of 1-3%, Wind vs. OMNI values. Notable differences pre-L1 for B.

df.drop(["Vomni_diff", "pomni_diff", "Bomni_diff"], axis=1, inplace=True)

df.to_csv("latest_results/wind_dataset.csv")
# Let's keep everything in here, but then for cleaned L1 dataset remove omni values.

#####################################################

df = pd.read_csv("latest_results/wind_dataset.csv")
df.Timestamp = pd.to_datetime(df.Timestamp)
df.set_index("Timestamp", inplace=True)
df.sort_index(inplace=True)

#### DATA CLEANING (subsetting and dealing with outliers)

df_l1 = df["2004-06-01":] 
# Wind has been at L1 continuously since June 2004: https://wind.nasa.gov/orbit.php 

# Few timestamps (0.1%) have ttc < 0 
# All of these have unusually large values for qk. Total of 5% have qk > -1.7
# 3 values also have ttu < 1

# Here I am removing all these values
# Removing NAs as well, this brings my total number of rows down to about 18,000
# It only slightly decreases the qk mean from -2.63 to -2.69, but it does
# remove very large Re_lt outliers, reducing the mean from 4,500,000 to 160,000
# It still leaves around 2% of rows where qk > qi

# Counting outliers using outlier flag columns
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
# df_l1_cleaned = df_l1_cleaned[df_l1_cleaned.ttu > 1] # not needed for L1 range

# Removing negative tci values (only 6, numerical issue with finding argmin)
df_l1_cleaned.loc[df_l1_cleaned.tci < 0, ["tci", "lambda_c_int"]] = np.nan

# Saving cleaned dataset
df_l1_cleaned.to_csv("latest_results/wind_dataset_l1_cleaned.csv", index=True)

# Saving correlations and summary statistics
corr_table = df_l1_cleaned.corr()
corr_table.to_csv("latest_results/wind_dataset_l1_cleaned_corr.csv")

key_vars = df_l1_cleaned[["lambda_c_fit", "lambda_c_int", "lambda_c_e", "lambda_t_raw", "qi", "qk", "fb", "lambda_t", "Re_lt", "Re_di", "Re_tb"]]

key_stats = key_vars.describe().round(2)
key_stats.to_csv("latest_results/wind_dataset_l1_cleaned_key_stats.csv")