#### ANALYSING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

######################################################


# TEMP CODE WHEN NEEDING TO MERGE FILES (i.e., did not run on all data at once)

# df_1 = pd.read_csv("data/processed/wind_database_1995_1998.csv")
# df_2 = pd.read_csv("data/processed/wind_database_1999_2007.csv")
# df_3 = pd.read_csv("data/processed/wind_database_2007_2022.csv")

# df_1 = df_1.set_index("Timestamp").sort_index()
# df_2 = df_2.set_index("Timestamp").sort_index()
# df_3 = df_3.set_index("Timestamp").sort_index()
# df_omni = df_1[["vsw", "p", "Bomni"]]

# # We have the entire OMNI data in each dataframe
# # We need to exclude it so it doesn't get added together during the following merging process
# # which takes into account the ragged transitions from one df to the next

# df_merged = pd.concat([df_1, df_2, df_3], verify_integrity=False)
# df_merged = df_merged.drop(["vsw", "p", "Bomni"], axis=1)
# df_merged.index.has_duplicates

# # # Can also check for duplicate timestamps during the concatentation with the following: 
# # #df_merged = pd.concat([df_1, df_2], verify_integrity=True)
# # #ValueError: Indexes have overlapping values

# df_merged = df_merged.groupby(df_merged.index).agg(sum)
# # Dealing with any resultant 0s from summing to NAs together
# df_merged = df_merged.replace(0, np.nan)

# df_merged.index = pd.to_datetime(df_merged.index)
# df_omni.index = pd.to_datetime(df_omni.index)
# df = utils.join_dataframes_on_timestamp(df_merged, df_omni)
# df.index.has_duplicates

# # # Checking merge (border between end of first file and start of second, with a ragged transition)
# # # df_merged_final["1998-12-30":"1999-01-03"]

# df.rename(columns={"tb":"fb"}, inplace=True)

# # df[["tcf", "ttc", "Re_di", "Re_lt", "Re_lt_u", "Re_tb"]].describe()
# # np.mean(df.Re_lt).round(-4)
# # np.mean(df.Re_di).round(-4)
# # np.mean(df.Re_tb).round(-4)

# # df[["di", "vsw", "ttk", "ttu", "ttc", "Re_di", "Re_lt", "Re_tb"]].describe().round(2)
# # # CHECK MAX VALS

# df.to_csv("data/processed/wind_database.csv")

######################################################

# df = pd.read_csv("data/processed/wind_omni_dataset.csv")
# df.Timestamp = pd.to_datetime(df.Timestamp)
# df.set_index("Timestamp", inplace=True)
# df.sort_index(inplace=True)

# #### DATA CLEANING (subsetting and dealing with outliers)

# df_l1 = df["2004-06-01":]

# Few timestamps (0.1%) have ttc < 0 
# All of these have unusually large values for qk. Total of 5% have qk > -1.7
# 3 values also have ttu < 1

# Here I am removing all these values
# Removing NAs as well, this brings my total number of rows down to about 18,000
# It only slightly decreases the qk mean from -2.63 to -2.69, but it does
# remove very large Re_lt outliers, reducing the mean from 4,500,000 to 160,000
# It still leaves around 2% of rows where qk > qi

# Counting outliers

# df_l1.loc[:, "small_ttu"] = 0 
# df_l1.loc[:, "qk > -1.7"] = 0
# df_l1.loc[:, "qk > qi"] = 0

# df_l1.loc[df_l1["ttu"] < 1, "small_ttu"] = 1
# df_l1.loc[df_l1["qk"] > -1.7, "qk > -1.7"] = 1
# df_l1.loc[df_l1["qk"] > df_l1["qi"], "qk > qi"] = 1

# df_l1[["small_ttu", "qk > -1.7", "qk > qi"]].mean()
# df_l1.groupby(["qk > -1.7", "qk > qi", "small_ttu"])[["small_ttu", "qk > -1.7", "qk > qi"]].value_counts()
# df_l1.drop(["small_ttu", "qk > -1.7", "qk > qi"], axis=1, inplace=True)

# # Removing outlier slope rows
# df_l1_cleaned = df_l1[df_l1.qk < -1.7]
# df_l1_cleaned = df_l1_cleaned[df_l1_cleaned.ttu > 1] # not needed for L1 range

# # Removing negative tci values (only 5, numerical issue with finding argmin)
# df_l1_cleaned.loc[df_l1_cleaned.tci < 0, ["tci", "lambda_c_int"]] = np.nan

# df_l1_cleaned.to_csv("data/processed/wind_dataset_l1_cleaned.csv", index=True)

# corr_table = df_l1_cleaned.corr()
# corr_table.to_csv("wind_dataset_l1_cleaned_corr.csv")

# key_vars = df_l1_cleaned[["lambda_c_fit", "lambda_c_int", "lambda_c_e", "lambda_t_raw", "qi", "qk", "fb", "lambda_t", "Re_lt", "Re_di", "Re_tb"]]

# key_stats = key_vars.describe().round(2)
# key_stats.to_csv("wind_dataset_l1_cleaned_key_stats.csv")

#######################################################
###################### PLOTTING #######################
#######################################################

df_l1_cleaned = pd.read_csv("data/processed/wind_dataset_l1_cleaned.csv")
df_l1_cleaned.Timestamp = pd.to_datetime(df_l1_cleaned.Timestamp)
df_l1_cleaned.set_index("Timestamp", inplace=True)
df_l1_cleaned.sort_index(inplace=True)
print(df_l1_cleaned.info())

## GETTING TIME PERIOD OF DATA

### HISTOGRAMS OF TAYLOR SCALES ###

fig, ax = plt.subplots(figsize=(6,3), constrained_layout=True)
sns.histplot(ax=ax, data=df_l1_cleaned.lambda_t_raw, log_scale=True, color = "cornflowerblue", label = "$\lambda_{T}^{extra}$")
sns.histplot(df_l1_cleaned.lambda_t, log_scale=True, color="green", label = "$\lambda_{T}$")
plt.axvline(df_l1_cleaned.lambda_t_raw.mean(), c="black")
plt.axvline(df_l1_cleaned.lambda_t.mean(), c="black")
plt.xlabel("Length (km)")
plt.xlim(500, 20000)
plt.text(df_l1_cleaned.lambda_t_raw.mean()*1.1, 600, "Mean = {:.0f}".format((df_l1_cleaned.lambda_t_raw.mean())))
plt.text(df_l1_cleaned.lambda_t.mean()/2, 600, "Mean = {:.0f}".format((df_l1_cleaned.lambda_t.mean())))
plt.legend()

plt.savefig("plots/final/taylor_overlapping_hist.pdf")
plt.show()

# PLOTS FOR ALL THREE RE APPROXIMATIONS

# unused helper functions

# Following fn courtesy of bnaecker on stackoverflow
# def plot_unity(xdata, ydata, **kwargs):
#     mn = min(xdata.min(), ydata.min())
#     mx = max(xdata.max(), ydata.max())
#     #mn = min(0, 1e4)
#     #mx = max(0, 1e7)
#     points = np.linspace(mn, mx, 100)
#     plt.gca().plot(points, points, color='k', marker=None,
#             linestyle='--', linewidth=1.0)
    
# def plot_unity_histplot(ax, **kwargs):
#     mn = min(1e1, 1e8)
#     mx = max(1e1, 1e8)
#     points = np.linspace(mn, mx, 100)
#     for i in np.arange(3):
#         ax[i].plot(points, points, color='k', marker=None,
#                 linestyle='--', linewidth=1.0)

# def meanfunc(x, ax=None, **kws):
#     #mean = np.mean(x)
#     #med = np.median(x)
#     x = pd.Series(x)
#     mean = x.mean().round(-4)
#     med = x.median().round(-4)
#     std = x.std().round(-4)
#     ax = ax or plt.gca()
#     ax.annotate(f'mean = \n{mean:.0f}', xy=(.1, .8), xycoords=ax.transAxes, size = 9)
#     ax.annotate(f'median = \n{med:.0f}', xy=(.1, .6), xycoords=ax.transAxes, size = 9)
#     ax.annotate(f'$\sigma$ = \n{std:.0f}', xy=(.1, .4), xycoords=ax.transAxes, size = 9)


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    x = pd.Series(x)
    y = pd.Series(y)
    rp = x.corr(y, "pearson")
    rs = x.corr(y, "spearman")
    ax = ax or plt.gca()
    ax.annotate(f'pearson: \n{rp:.2f}', xy=(.65, .3), xycoords=ax.transAxes, size=7)
    ax.annotate(f'spearman: \n{rs:.2f}', xy=(.65, .1), xycoords=ax.transAxes, size=7)

# Create the plot
fig = plt.figure(figsize=(7, 3))
grid = fig.add_gridspec(4, 3, hspace=0)

# Define the axes for the plot
ax_marg_x_0 = fig.add_subplot(grid[0, 0:1])
ax_joint_0 = fig.add_subplot(grid[1:4, 0:1])
ax_marg_x_1 = fig.add_subplot(grid[0, 1:2])
ax_joint_1 = fig.add_subplot(grid[1:4, 1:2])
ax_marg_x_2 = fig.add_subplot(grid[0, 2:3])
ax_joint_2 = fig.add_subplot(grid[1:4, 2:3])

# Create the plot using seaborn's jointplot function
sns.kdeplot(data=df_l1_cleaned, x="Re_tb", ax=ax_marg_x_0, log_scale=True)
sns.kdeplot(data=df_l1_cleaned, x="Re_di", ax=ax_marg_x_1, log_scale=True)
sns.kdeplot(data=df_l1_cleaned, x="Re_lt", ax=ax_marg_x_2, log_scale=True)

sns.histplot(ax = ax_joint_0, data=df_l1_cleaned, x="Re_tb", y="Re_lt", log_scale=True)
corrfunc(df_l1_cleaned["Re_tb"], df_l1_cleaned["Re_lt"], ax_joint_0)
ax_joint_0.set_xlabel("$Re_{tb}$")
ax_joint_0.set_ylabel("$Re_{\lambda_T}$")

sns.histplot(ax = ax_joint_1, data=df_l1_cleaned, x="Re_di", y="Re_tb", log_scale=True)
corrfunc(df_l1_cleaned["Re_di"], df_l1_cleaned["Re_tb"], ax_joint_1)
ax_joint_1.set_xlabel("$Re_{di}$")
ax_joint_1.set_ylabel("$Re_{tb}$")

sns.histplot(ax = ax_joint_2, data=df_l1_cleaned, x="Re_lt", y="Re_di", log_scale=True)
corrfunc(df_l1_cleaned["Re_lt"], df_l1_cleaned["Re_di"], ax_joint_2)
ax_joint_2.set_xlabel("$Re_{\lambda_T}$")
ax_joint_2.set_ylabel("$Re_{di}$")

for ax in [ax_marg_x_0, ax_marg_x_1, ax_marg_x_2]:
    ax.set_ylim(0, 1.2)
    ax.set_xlim(1e3, 1e7)
    ax.axis('off')

for ax in [ax_joint_0, ax_joint_1, ax_joint_2]:
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(1e3, 1e7)
    ax.plot([1e3, 1e7], [1e3, 1e7], linestyle='--', linewidth=1.0, c = "black")
    ax.plot([1e3, 1e7], [1e3, 1e7], linestyle='--', linewidth=1.0, c = "black")
    ax.plot([1e3, 1e7], [1e3, 1e7], linestyle='--', linewidth=1.0, c = "black")
    ax.set_xticks([1e4,1e6])
    ax.set_yticks([1e4,1e6])
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.minorticks_off()
    ax.grid()

ax_joint_0.tick_params(direction='in')
ax_joint_1.tick_params(direction='in', labelleft=False)
ax_joint_2.tick_params(direction='in', labelleft=False)

fig.tight_layout()

# Save/show the plot
plt.savefig("plots/final/re_panels.pdf")
plt.show()


# NOW FOR CORR SCALES

# Create the plot
fig = plt.figure(figsize=(7, 3))
grid = fig.add_gridspec(4, 3, hspace=0)

# Define the axes for the plot
ax_marg_x_0 = fig.add_subplot(grid[0, 0:1])
ax_joint_0 = fig.add_subplot(grid[1:4, 0:1])
ax_marg_x_1 = fig.add_subplot(grid[0, 1:2])
ax_joint_1 = fig.add_subplot(grid[1:4, 1:2])
ax_marg_x_2 = fig.add_subplot(grid[0, 2:3])
ax_joint_2 = fig.add_subplot(grid[1:4, 2:3])

# Create the plot using seaborn's jointplot function
sns.kdeplot(data=df_l1_cleaned, x="lambda_c_e", ax=ax_marg_x_0, log_scale=True)
sns.kdeplot(data=df_l1_cleaned, x="lambda_c_fit", ax=ax_marg_x_1, log_scale=True)
sns.kdeplot(data=df_l1_cleaned, x="lambda_c_int", ax=ax_marg_x_2, log_scale=True)

sns.histplot(ax = ax_joint_0, data=df_l1_cleaned, x="lambda_c_fit", y="lambda_c_e", log_scale=True)
corrfunc(df_l1_cleaned["lambda_c_fit"], df_l1_cleaned["lambda_c_e"], ax_joint_0)
ax_joint_0.set_xlabel("$\lambda_{C}^{fit}$ (km)")
ax_joint_0.set_ylabel("$\lambda_{C}^{1/e}$ (km)")

sns.histplot(ax = ax_joint_1, data=df_l1_cleaned, x="lambda_c_e", y="lambda_c_int", log_scale=True)
corrfunc(df_l1_cleaned["lambda_c_e"], df_l1_cleaned["lambda_c_int"], ax_joint_1)
ax_joint_1.set_xlabel("$\lambda_{C}^{1/e}$ (km)")
ax_joint_1.set_ylabel("$\lambda_{C}^{int}$ (km)")

sns.histplot(ax = ax_joint_2, data=df_l1_cleaned, x="lambda_c_int", y="lambda_c_fit", log_scale=True)
corrfunc(df_l1_cleaned["lambda_c_int"], df_l1_cleaned["lambda_c_fit"], ax_joint_2)
ax_joint_2.set_xlabel("$\lambda_{C}^{int}$ (km)")
ax_joint_2.set_ylabel("$\lambda_{C}^{fit}$ (km)")

for ax in [ax_marg_x_0, ax_marg_x_1, ax_marg_x_2]:
    ax.set_ylim(0, 2.2)
    ax.set_xlim(1e5, 1e7) # Annoying adjustment of xlim upper needed to make the gridlines look nice
    ax.axis('off')

for ax in [ax_joint_0, ax_joint_1, ax_joint_2]:
    ax.set_xlim(1e5, 1e7)
    ax.set_ylim(1e5, 1e7)
    ax.plot([1e5, 1e7], [1e5, 1e7], linestyle='--', linewidth=1.0, c = "black")
    ax.plot([1e5, 1e7], [1e5, 1e7], linestyle='--', linewidth=1.0, c = "black")
    ax.plot([1e5, 1e7], [1e5, 1e7], linestyle='--', linewidth=1.0, c = "black")
    # ax.set_xticks([1e4,1e6])
    # ax.set_yticks([1e4,1e6])
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.minorticks_off()
    ax.grid()

ax_joint_0.tick_params(direction='in')
ax_joint_1.tick_params(direction='in', labelleft=False)
ax_joint_2.tick_params(direction='in', labelleft=False)

fig.tight_layout()

# Save/show the plot
plt.savefig("plots/final/corr_scale_panels.pdf")
plt.show()


# x0, x1 = g.ax_joint.get_xlim()
# y0, y1 = g.ax_joint.get_ylim()
# lims = [max(x0, y0), min(x1, y1)]
# g.ax_joint.plot(lims, lims, '-r')

# TIME SERIES OF RE

# df_l1_cleaned.Timestamp = pd.to_datetime(df_l1_cleaned.Timestamp)
# df_l1_cleaned_1995 = df_l1_cleaned[(df_l1_cleaned.Timestamp > "1997-01-01") & (df_l1_cleaned.Timestamp < "1998-01-01")]

# # Min-max normalisation
# df_l1_cleaned_1995["Re_di_norm"] = (df_l1_cleaned_1995["Re_di"]-df_l1_cleaned_1995["Re_di"].min())/(df_l1_cleaned_1995["Re_di"].max()-df_l1_cleaned_1995["Re_di"].min())
# df_l1_cleaned_1995["Re_lt_norm"] = (df_l1_cleaned_1995["Re_lt"]-df_l1_cleaned_1995["Re_lt"].min())/(df_l1_cleaned_1995["Re_lt"].max()-df_l1_cleaned_1995["Re_lt"].min())
# df_l1_cleaned_1995["Re_tb_norm"] = (df_l1_cleaned_1995["Re_tb"]-df_l1_cleaned_1995["Re_tb"].min())/(df_l1_cleaned_1995["Re_tb"].max()-df_l1_cleaned_1995["Re_tb"].min())

# sns.lineplot(data=df_l1_cleaned_1995, x="Timestamp", y="Re_di_norm", label="Re_di")
# sns.lineplot(data=df_l1_cleaned_1995, x="Timestamp", y="Re_lt_norm", label="Re_lt")
# sns.lineplot(data=df_l1_cleaned_1995, x="Timestamp", y="Re_tb_norm", label="Re_tb")

# plt.semilogy()

# plt.show()