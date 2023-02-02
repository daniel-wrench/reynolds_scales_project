#### ANALYSING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import src.utils as utils

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

# # # TEMP CODE FOR CALCULATING REYNOLDS NUMBERS 

# df["Re_lt"] = (df["tcf"]/df["ttc"])**2
# df["Re_lt_u"] = (df["tcf"]/df["ttu"])**2
# df["Re_di"] = ((df["tcf"]*df["vsw"])/df["di"])**(4/3)

# df.rename(columns={"tb":"fb"}, inplace=True)
# df["tb"] = 1/((2*np.pi)*df["fb"])
# df["Re_tb"] = ((df["tcf"]/df["tb"]))**(4/3)

# # df[["tcf", "ttc", "Re_di", "Re_lt", "Re_lt_u", "Re_tb"]].describe()
# # np.mean(df.Re_lt).round(-4)
# # np.mean(df.Re_di).round(-4)
# # np.mean(df.Re_tb).round(-4)

# # df[["di", "vsw", "ttk", "ttu", "ttc", "Re_di", "Re_lt", "Re_tb"]].describe().round(2)
# # # CHECK MAX VALS

# df.to_csv("data/processed/wind_database.csv")

######################################################

df = pd.read_csv("data/processed/wind_database.csv")

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


# Removing outliers 
df_cleaned = df[df.qk < -1.7]
df_cleaned = df_cleaned[df_cleaned.ttu > 1] # 3 values

df_cleaned.Re_lt_u.describe()

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
df_no_na = df.dropna()
print(df_no_na.Timestamp.min(), df_no_na.Timestamp.max())

corr_mat = df_no_na.corr()
corr_mat.to_csv("corr_mat.csv")
# Save these extreme cases as case studies (as with strange slopes), but exclude from main statistical analysis

### SCATTERPLOTS AND HISTOGRAMS OF TAYLOR SCALES ###

g = sns.JointGrid(
    data=df_cleaned, 
    x="ttu_km", 
    y="ttc_km"#, 
    #xlim = (1e3, 1e7), 
    #ylim=(1e3, 1e7)
    )

g.ax_joint.set(xscale = "log", yscale="log")

# Create an inset legend for the histogram colorbar
#cax = g.figure.add_axes([.15, .55, .02, .2])

# Add the joint and marginal histogram plots
g.plot_joint(
    sns.histplot  #,
    #cmap="light:#03012d", 
    #pmax=.8, 
    #cbar=True
)
g.plot_marginals(
    sns.histplot  #, or kdeplot if you want densities 
    # element="step",
    # color="#03012d"
)

# Draw a line of x=y 
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, '--', c="black")

#Plot the means
# g.ax_joint.axvline(np.mean(df_cleaned.ttu_km))
# g.ax_joint.axhline(np.mean(df_cleaned.ttc_km))

#Draw a regression line
#g.plot_joint(sns.regplot, scatter=False, ci=False)

#Also add correlation + log fit regression line

# fit = np.polyfit(np.array(df_cleaned_clean["Re_di"]), np.array(df_cleaned_clean["Re_lt"]), deg=1)
# g.ax_joint.plot(df_cleaned_clean[["Re_di"]], df_cleaned_clean["Re_di"]*fit[0]+fit[1], '-b')

# df_cleaned_clean = df_cleaned.dropna()
# reg = LinearRegression().fit(df_cleaned_clean[["Re_di"]], df_cleaned_clean[["Re_lt"]])
# Re_lt_predict = reg.predict(df_cleaned_clean[["Re_di"]])

## The coefficients
# print("Coefficients: \n", reg.coef_)
# print(reg.intercept_)

# from sklearn.metrics import mean_squared_error, r2_score
# print("Mean squared error: %.2f" % mean_squared_error(df_cleaned_clean[["Re_lt"]], Re_lt_predict))
# The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(df_cleaned_clean[["Re_lt"]], Re_lt_predict))

g.set_axis_labels(xlabel = "$\lambda_{TS}^{extra}$ (km)", ylabel = "$\lambda_{TS}$ (km)")
plt.savefig("plots/final/taylor_bivariate.png")
plt.show()

sns.histplot(df_cleaned.ttu_km, log_scale=True, label = "$\lambda_{TS}^{extra}$ (km)")
sns.histplot(df_cleaned.ttc_km, log_scale=True, color="orange", label = "$\lambda_{TS}$ (km)")
plt.xlabel("Length (km)")
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

f = sns.PairGrid(reynolds, diag_sharey=False, corner=False)
f.map_lower(sns.histplot, log_scale=True)
f.map_lower(plot_unity)
f.map_lower(corrfunc)

#f.map_lower(sns.regplot, scatter=False)
f.map_diag(sns.kdeplot, log_scale=True)
f.map_diag(meanfunc)

# f.axes[0,0].set_xlim(1e1, 1e8)
# f.axes[0,0].text(.8, .85, "Mean: 0.5", transform=f.axes[0,0].transAxes, fontweight="bold")
# f.axes[1,1].text(.8, .85, "Mean: 0.5", transform=f.axes[1,1].transAxes, fontweight="bold")
# f.axes[2,2].text(.8, .85, "Mean: 0.5", transform=f.axes[2,2].transAxes, fontweight="bold")

#f.axes[2,2].set_xlabel("$Re_{lt}$")

f.axes[0,0].set_xlim(1e1, 1e8)
f.axes[0,0].set_ylim(1e1, 1e8)

f.axes[1,1].set_xlim(1e1, 1e8)
f.axes[1,1].set_ylim(1e1, 1e8)

f.axes[2,2].set_xlim(1e1, 1e8)
f.axes[2,2].set_ylim(1e1, 1e8)

plt.savefig("plots/final/re_matrix.pdf")
#plt.tight_layout()
plt.show()






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

f = sns.PairGrid(df_cleaned[["tce_km", "tcf_km", "tci_km"]], diag_sharey=False, corner=False)
f.map_lower(sns.histplot, log_scale=True)
f.map_lower(plot_unity)
f.map_lower(corrfunc)

#f.map_lower(sns.regplot, scatter=False)
f.map_diag(sns.kdeplot, log_scale=True)
f.map_diag(meanfunc)

# f.axes[0,0].set_xlim(1e1, 1e8)
# f.axes[0,0].text(.8, .85, "Mean: 0.5", transform=f.axes[0,0].transAxes, fontweight="bold")
# f.axes[1,1].text(.8, .85, "Mean: 0.5", transform=f.axes[1,1].transAxes, fontweight="bold")
# f.axes[2,2].text(.8, .85, "Mean: 0.5", transform=f.axes[2,2].transAxes, fontweight="bold")

#f.axes[2,2].set_xlabel("$Re_{lt}$")

f.axes[0,0].set_xlim(1e4, 1e7)
f.axes[0,0].set_ylim(1e4, 1e7)

f.axes[1,1].set_xlim(1e4, 1e7)
f.axes[1,1].set_ylim(1e4, 1e7)

f.axes[2,2].set_xlim(1e4, 1e7)
f.axes[2,2].set_ylim(1e4, 1e7)

plt.savefig("plots/final/corr_scale_matrix.pdf")
#plt.tight_layout()
plt.show()


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



# Checking weird stuff
df = pd.read_csv("data/processed/wind_database.csv")
df.drop(df.columns[0], axis = 1, inplace=True)
df.Timestamp = pd.to_datetime(df.Timestamp)
df.set_index("Timestamp", inplace=True)
check = df.loc["2005":"2015",["ni", "di", "Re_di"]]

check.ni_norm = (check.ni-check.ni.min())/(check.ni.max()-check.ni.min())
check.di_norm = (check.di-check.di.min())/(check.di.max()-check.di.min())
#check.Re_di = (check.Re_di-check.Re_di.min())/(check.Re_di.max()-check.Re_di.min())

fig, ax = plt.subplots(1,3)
ax[1].plot(check.di, label = "di")
ax[1].plot(check.ni, label = "ni")
ax[0].plot(check.di_norm, label = "di_norm")
ax[0].plot(check.ni_norm, label = "ni_norm")
ax[2].plot(check.Re_di, label = "Re_di")
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()

# Show to Tulasi
# Note this will also be dragging down the Re_di values over this period
