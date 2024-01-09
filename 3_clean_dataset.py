#####################################################

# SUBSETTING TO L1 AND GETTING RID OF OUTLIERS

import numpy as np
import pandas as pd
import src.params as params

df = pd.read_csv("wind_dataset.csv")
df.Timestamp = pd.to_datetime(df.Timestamp)
df.set_index("Timestamp", inplace=True)
df.sort_index(inplace=True)

#### DATA CLEANING (subsetting and dealing with outliers)

df_l1 = df["2004-06-01":].copy()
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
df_l1_cleaned.to_csv("wind_dataset_l1_" + params.int_size + "_cleaned.csv", index=True)

# Saving correlations and summary statistics
corr_table = df_l1_cleaned.corr()
corr_table.to_csv("wind_dataset_" + params.int_size + "_l1_cleaned_corr.csv")

stats = df_l1_cleaned.describe().round(2)
stats.to_csv("wind_dataset_" + params.int_size + "_l1_cleaned_stats.csv")