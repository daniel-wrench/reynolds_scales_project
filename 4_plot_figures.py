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

df = pd.read_csv("data/processed/wind_database.csv")

# Calculating analytically-derived variables
# Using ne in place of ni due to issues with wind ni data.

df["rhoe"] = (2.38e-5)*(df["Te"]**(1/2))*((df["Bwind"]*1e-5)**-1)  # Electron gyroradius
df["rhoi"] = (1.02e-3)*(df["Ti"]**(1/2))*((df["Bwind"]*1e-5)**-1) # Ion gyroradius
df["de"] = (5.31)*(df["ne"]**(-1/2)) # Electron inertial length
df["di"] = (2.28e2)*(df["ne"]**(-1/2)) # Ion inertial length
df["betae"] = (4.03e-11)*df["ne"]*df["Te"]*((df["Bwind"]*1e-5)**-2) # Electron plasma beta
# df["betai"] = (4.03e-11)*df["ni"]*df["Ti"]*((df["Bwind"]*1e-5)**-2) # Ion plasma beta
df["va"] = (2.18e6)*(df["ne"]**(-1/2))*(df["Bwind"]*1e-5) # Alfven speed
df["ld"] = (7.43e-3)*(df["Te"]**(1/2))*(df["ne"]**(-1/2)) # Debye length

#### DATA CLEANING (dealing with outliers)

# Few timestamps (0.1%) have ttc < 0 
# All of these have unusually large values for qk. Total of 5% have qk > -1.7
# 3 values also have ttu < 1

# Here I am removing all these values
# Removing NAs as well, this brings my total number of rows down to about 18,000
# It only slightly decreases the qk mean from -2.63 to -2.69, but it does
# remove very large Re_lt outliers, reducing the mean from 4,500,000 to 160,000
# It still leaves around 2% of rows where qk > qi


# Tabulating counts and percentages of these outliers

# df.loc[:, "negative_ttc"] = 0 
# df.loc[:, "qk > -1.7"] = 0
# df.loc[:, "qk > qi"] = 0

# df.loc[df["ttc"] < 0, "negative_ttc"] = 1
# df.loc[df["qk"] > -1.7, "qk > -1.7"] = 1
# df.loc[df["qk"] > df["qi"], "qk > qi"] = 1

# df[["negative_ttc", "qk > -1.7", "qk > qi"]].mean()
# df.groupby(["qk > -1.7", "qk > qi", "negative_ttc"])[["negative_ttc", "qk > -1.7", "qk > qi"]].value_counts()

# Removing all these rows due to flow-on effects
df_cleaned = df[df.qk < -1.7]
df_cleaned = df_cleaned[df_cleaned.ttu > 1] # 3 values

# Removing other outliers 
df_cleaned.loc[df_cleaned.tci <0, "tci"] = np.nan

df_cleaned.to_csv("data/processed/wind_database_cleaned.csv", index=False)

## CONVERTING SCALES FROM TIME TO DISTANCE

df_cleaned['ttk_km'] = df_cleaned["ttk"]*df_cleaned["vsw"]
df_cleaned['ttu_km'] = df_cleaned["ttu"]*df_cleaned["vsw"]
df_cleaned['ttc_km'] = df_cleaned["ttc"]*df_cleaned["vsw"]
df_cleaned['tce_km'] = df_cleaned["tce"]*df_cleaned["vsw"]
df_cleaned['tcf_km'] = df_cleaned["tcf"]*df_cleaned["vsw"]
df_cleaned['tci_km'] = df_cleaned["tci"]*df_cleaned["vsw"]

corr_table = df_cleaned.corr()

key_vars = df_cleaned[["tcf_km", "tci_km", "tce_km", "ttk_km", "ttu_km", "qi", "qk", "fb", "ttc_km", "Re_lt", "Re_lt_u", "Re_di", "Re_tb"]]

stats = key_vars.describe().round(2)
stats.to_csv("wind_database_summary_stats_cleaned.csv")

# key_vars.hist(bins=100)
# plt.tight_layout()
# plt.show()

# df.ttu.quantile(0.1)

## GETTING TIME PERIOD OF DATA

df_no_na = df_cleaned.dropna()
print(df_no_na.Timestamp.min(), df_no_na.Timestamp.max())

corr_mat = df_no_na.corr()
corr_mat.to_csv("cleaned_corr_mat.csv")
# Save these extreme cases as case studies (as with strange slopes), but exclude from main statistical analysis



df_cleaned = pd.read_csv("data/processed/wind_database_cleaned.csv")




### HISTOGRAMS OF TAYLOR SCALES ###

fig, ax = plt.subplots(figsize=(6,3), constrained_layout=True)
sns.histplot(ax=ax, data=df_cleaned.ttu_km, log_scale=True, color = "cornflowerblue", label = "$\lambda_{TS}^{extra}$")
sns.histplot(df_cleaned.ttc_km, log_scale=True, color="green", label = "$\lambda_{TS}$")
plt.axvline(df_cleaned.ttu_km.mean(), c="black")
plt.axvline(df_cleaned.ttc_km.mean(), c="black")
plt.xlabel("Length (km)")
plt.xlim(500, 20000)
plt.text(df_cleaned.ttu_km.mean()*1.1, 800, "Mean = {:.0f}".format((df_cleaned.ttu_km.mean())))
plt.text(df_cleaned.ttc_km.mean()/2, 800, "Mean = {:.0f}".format((df_cleaned.ttc_km.mean())))
plt.legend()

plt.savefig("plots/final/taylor_overlapping_hist.pdf")

# PLOTS FOR ALL THREE RE APPROXIMATIONS

reynolds = df_cleaned[["Re_di", "Re_tb", "Re_lt"]]

# Following fn courtesy of bnaecker on stackoverflow
def plot_unity(xdata, ydata, **kwargs):
    #mn = min(xdata.min(), ydata.min())
    #mx = max(xdata.max(), ydata.max())
    mn = min(1e1, 1e8)
    mx = max(1e1, 1e8)
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='k', marker=None,
            linestyle='--', linewidth=1.0)

def meanfunc(x, ax=None, **kws):
    #mean = np.mean(x)
    #med = np.median(x)
    x = pd.Series(x)
    mean = x.mean().round(-4)
    med = x.median().round(-4)
    std = x.std().round(-4)
    ax = ax or plt.gca()
    ax.annotate(f'mean = \n{mean:.0f}', xy=(.1, .8), xycoords=ax.transAxes, size = 9)
    ax.annotate(f'median = \n{med:.0f}', xy=(.1, .6), xycoords=ax.transAxes, size = 9)
    ax.annotate(f'$\sigma$ = \n{std:.0f}', xy=(.1, .4), xycoords=ax.transAxes, size = 9)

# f = sns.PairGrid(reynolds, diag_sharey=False, corner=True, despine=False)
# f.map_lower(sns.histplot, log_scale=True)
# f.map_lower(plot_unity)
# f.map_lower(corrfunc)

# #f.map_lower(sns.regplot, scatter=False)
# f.map_diag(sns.kdeplot, log_scale=True)
# f.map_diag(meanfunc)

# # f.axes[0,0].set_xlim(1e1, 1e8)
# # f.axes[0,0].text(.8, .85, "Mean: 0.5", transform=f.axes[0,0].transAxes, fontweight="bold")
# # f.axes[1,1].text(.8, .85, "Mean: 0.5", transform=f.axes[1,1].transAxes, fontweight="bold")
# # f.axes[2,2].text(.8, .85, "Mean: 0.5", transform=f.axes[2,2].transAxes, fontweight="bold")

# #f.axes[2,2].set_xlabel("$Re_{lt}$")

# f.axes[0,0].set_xlim(1e1, 1e8)
# f.axes[0,0].set_ylim(1e1, 1e8)

# f.axes[1,1].set_xlim(1e1, 1e8)
# f.axes[1,1].set_ylim(1e1, 1e8)

# f.axes[2,2].set_xlim(1e1, 1e8)
# f.axes[2,2].set_ylim(1e1, 1e8)

# #plt.savefig("plots/final/re_matrix.pdf")
# #plt.tight_layout()
# plt.show()

def plot_unity_histplot(ax, **kwargs):
    mn = min(1e1, 1e8)
    mx = max(1e1, 1e8)
    points = np.linspace(mn, mx, 100)
    for i in np.arange(3):
        ax[i].plot(points, points, color='k', marker=None,
                linestyle='--', linewidth=1.0)

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    x = pd.Series(x)
    y = pd.Series(y)
    rp = x.corr(y, "pearson")
    rs = x.corr(y, "spearman")
    ax = ax or plt.gca()
    ax.annotate(f'pearson: \n{rp:.2f}', xy=(.65, .3), xycoords=ax.transAxes, size=7)
    ax.annotate(f'spearman: \n{rs:.2f}', xy=(.65, .1), xycoords=ax.transAxes, size=7)

fig, ax = plt.subplots(1, 3, figsize=(7,2.5), constrained_layout=True, sharex=True)
sns.histplot(ax = ax[0], data=df_cleaned, x="Re_tb", y="Re_lt", log_scale=True)
corrfunc(df_cleaned["Re_tb"], df_cleaned["Re_lt"], ax[0])
sns.histplot(ax = ax[1], data=df_cleaned, x="Re_tb", y="Re_di", log_scale=True)
corrfunc(df_cleaned["Re_tb"], df_cleaned["Re_di"], ax[1])
sns.histplot(ax = ax[2], data=df_cleaned, x="Re_lt", y="Re_di", log_scale=True)
corrfunc(df_cleaned["Re_lt"], df_cleaned["Re_di"], ax[2])

ax[0].set_xlim(1e3, 1e7)
ax[0].set_ylim(1e3, 1e7)
ax[1].set_ylim(1e3, 1e7)
ax[2].set_ylim(1e3, 1e7)
plot_unity_histplot(ax)
#plt.show()

plt.savefig("plots/final/re_panels.pdf")

# NOW FOR CORR SCALES

# Following fn courtesy of bnaecker on stackoverflow
def plot_unity(xdata, ydata, **kwargs):
    mn = min(xdata.min(), ydata.min())
    mx = max(xdata.max(), ydata.max())
    #mn = min(0, 1e4)
    #mx = max(0, 1e7)
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(points, points, color='k', marker=None,
            linestyle='--', linewidth=1.0)

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    x = pd.Series(x)
    y = pd.Series(y)
    rp = x.corr(y, "pearson")
    rs = x.corr(y, "spearman")
    ax = ax or plt.gca()
    ax.annotate(f'pearson = \n{rp:.2f}', xy=(.1, .8), xycoords=ax.transAxes, size = 9)
    ax.annotate(f'spearman = \n{rs:.2f}', xy=(.1, .6), xycoords=ax.transAxes, size = 9)

def meanfunc(x, ax=None, **kws):
    #mean = np.mean(x)
    #med = np.median(x)
    x = pd.Series(x)
    mean = x.mean().round(-4)
    med = x.median().round(-4)
    std = x.std().round(-4)
    ax = ax or plt.gca()
    ax.annotate(f'mean = \n{mean:.0f}', xy=(.1, .8), xycoords=ax.transAxes, size = 9)
    ax.annotate(f'median = \n{med:.0f}', xy=(.1, .6), xycoords=ax.transAxes, size = 9)
    ax.annotate(f'$\sigma$ = \n{std:.0f}', xy=(.1, .4), xycoords=ax.transAxes, size = 9)


fig, ax = plt.subplots(1, 3, figsize=(7,2.5), constrained_layout=True, sharex=True)
sns.histplot(ax = ax[0], data=df_cleaned, x="tce_km", y="tcf_km", log_scale=True)
corrfunc(df_cleaned["tce"], df_cleaned["tcf"], ax[0])
sns.histplot(ax = ax[1], data=df_cleaned, x="tce_km", y="tci_km", log_scale=True)
corrfunc(df_cleaned["tce"], df_cleaned["tci"], ax[1])
sns.histplot(ax = ax[2], data=df_cleaned, x="tcf_km", y="tci_km", log_scale=True)
corrfunc(df_cleaned["tcf"], df_cleaned["tci"], ax[2])
ax[0].set_xlim(1e5, 1e7)
ax[0].set_ylim(1e5, 1e7)
ax[1].set_ylim(1e5, 1e7)
ax[2].set_ylim(1e5, 1e7)
plot_unity_histplot(ax)
plt.savefig("plots/final/corr_scale_panels.pdf")
#plt.tight_layout()
#plt.show()


# x0, x1 = g.ax_joint.get_xlim()
# y0, y1 = g.ax_joint.get_ylim()
# lims = [max(x0, y0), min(x1, y1)]
# g.ax_joint.plot(lims, lims, '-r')

# TIME SERIES OF RE

# df_cleaned.Timestamp = pd.to_datetime(df_cleaned.Timestamp)
# df_cleaned_1995 = df_cleaned[(df_cleaned.Timestamp > "1997-01-01") & (df_cleaned.Timestamp < "1998-01-01")]

# # Min-max normalisation
# df_cleaned_1995["Re_di_norm"] = (df_cleaned_1995["Re_di"]-df_cleaned_1995["Re_di"].min())/(df_cleaned_1995["Re_di"].max()-df_cleaned_1995["Re_di"].min())
# df_cleaned_1995["Re_lt_norm"] = (df_cleaned_1995["Re_lt"]-df_cleaned_1995["Re_lt"].min())/(df_cleaned_1995["Re_lt"].max()-df_cleaned_1995["Re_lt"].min())
# df_cleaned_1995["Re_tb_norm"] = (df_cleaned_1995["Re_tb"]-df_cleaned_1995["Re_tb"].min())/(df_cleaned_1995["Re_tb"].max()-df_cleaned_1995["Re_tb"].min())

# sns.lineplot(data=df_cleaned_1995, x="Timestamp", y="Re_di_norm", label="Re_di")
# sns.lineplot(data=df_cleaned_1995, x="Timestamp", y="Re_lt_norm", label="Re_lt")
# sns.lineplot(data=df_cleaned_1995, x="Timestamp", y="Re_tb_norm", label="Re_tb")

# plt.semilogy()

# plt.savefig("plots/re_time_series.png")
# plt.show()