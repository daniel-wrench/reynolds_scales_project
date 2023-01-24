#### ANALYSING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import src.utils as utils

# TEMP CODE WHEN NEEDING TO MERGE FILES (i.e., did not run on all data at once)

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

# # # Checking merge (border between end of first file and start of second, with a ragged transition)
# # # df_merged_final["1998-12-30":"1999-01-03"]

# # # TEMP CODE FOR CALCULATING REYNOLDS NUMBERS 

df["Re_lt"] = (df["tcf"]/df["ttc"])**2
df["Re_di"] = ((df["tcf"]*df["vsw"])/df["di"])**(4/3)

df.rename(columns={"tb":"fb"}, inplace=True)
df["tb"] = 1/((2*np.pi)*df["fb"])
df["Re_tb"] = ((df["tcf"]/df["tb"]))**(4/3)

# # df[["tcf", "ttc", "Re_di", "Re_lt", "Re_lt_u", "Re_tb"]].describe()
# # np.mean(df.Re_lt).round(-4)
# # np.mean(df.Re_di).round(-4)
# # np.mean(df.Re_tb).round(-4)

# # df[["di", "vsw", "ttk", "ttu", "ttc", "Re_di", "Re_lt", "Re_tb"]].describe().round(2)
# # # CHECK MAX VALS

df.to_csv("data/processed/wind_database.csv")

######################################################

df = pd.read_csv("data/processed/wind_database.csv")

#### DATA CLEANING (dealing with outliers)

# Drop points where ratio > 100 and ratio < 0.01 
# or as below, drop very large values (skewed distributions)

df_cleaned = df
#df_cleaned = df[df["ttc"] > 0] 
#df_cleaned = df_cleaned[df_cleaned.qi > df_cleaned.qk]

## CONVERTING SCALES FROM TIME TO DISTANCE

df_cleaned['ttk_km'] = df_cleaned["ttk"]*df_cleaned["vsw"]
df_cleaned['ttu_km'] = df_cleaned["ttu"]*df_cleaned["vsw"]
df_cleaned['ttc_km'] = df_cleaned["ttc"]*df_cleaned["vsw"]
df_cleaned['tce_km'] = df_cleaned["tce"]*df_cleaned["vsw"]
df_cleaned['tcf_km'] = df_cleaned["tcf"]*df_cleaned["vsw"]
df_cleaned['tci_km'] = df_cleaned["tci"]*df_cleaned["vsw"]

stats = df_cleaned[["tcf_km", "tci_km", "tce_km", "ttk_km", "ttu_km", "qi", "qk", "ttc_km", "Re_lt", "Re_di", "Re_tb"]].describe().round(2)
stats.to_csv("wind_database_summary_stats.csv")
## GETTING TIME PERIOD OF DATA

df_no_na = df.dropna()
print(df_no_na.Timestamp.min(), df_no_na.Timestamp.max())

corr_mat = df_no_na.corr()
corr_mat.to_csv("corr_mat.csv")
# Save these extreme cases as case studies (as with strange slopes), but exclude from main statistical analysis

### SCATTERPLOTS AND HISTOGRAMS OF RE ###

g = sns.JointGrid(
    data=df_cleaned, 
    x="Re_di", 
    y="Re_lt", 
    xlim = (1e3, 1e7), 
    ylim=(1e3, 1e7)
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
g.ax_joint.plot(lims, lims, '-r')

# Plot the means
# g.ax_joint.axhline(np.mean(df_cleaned.Re_lt))
# g.ax_joint.axvline(np.mean(df_cleaned.Re_di))

# Draw a regression line
# g.plot_joint(sns.regplot, scatter=False, ci=False)

# Also add correlation + log fit regression line

# fit = np.polyfit(np.array(df_cleaned_clean["Re_di"]), np.array(df_cleaned_clean["Re_lt"]), deg=1)
# g.ax_joint.plot(df_cleaned_clean[["Re_di"]], df_cleaned_clean["Re_di"]*fit[0]+fit[1], '-b')

# df_cleaned_clean = df_cleaned.dropna()
# reg = LinearRegression().fit(df_cleaned_clean[["Re_di"]], df_cleaned_clean[["Re_lt"]])
# Re_lt_predict = reg.predict(df_cleaned_clean[["Re_di"]])

# The coefficients
#print("Coefficients: \n", reg.coef_)
#print(reg.intercept_)

# from sklearn.metrics import mean_squared_error, r2_score
# print("Mean squared error: %.2f" % mean_squared_error(df_cleaned_clean[["Re_lt"]], Re_lt_predict))
# The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(df_cleaned_clean[["Re_lt"]], Re_lt_predict))
# plt.savefig("plots/re_bivariate.png")
plt.show()

# PLOTS FOR ALL THREE RE APPROXIMATIONS

df_cleaned_re = df_cleaned[["Re_di", "Re_tb", "Re_lt"]]
df_cleaned_re = df_cleaned_re.dropna()
f = sns.PairGrid(df_cleaned_re, diag_sharey=False, corner=False)
f.map_lower(sns.histplot, log_scale=True)
#f.map_lower(sns.regplot, scatter=False)
f.map_diag(sns.kdeplot, log_scale=True)
plt.savefig("plots/re_trivariate.png")
plt.show()

# TIME SERIES OF RE

df_cleaned.Timestamp = pd.to_datetime(df_cleaned.Timestamp)
df_cleaned_1995 = df_cleaned[(df_cleaned.Timestamp > "1997-01-01") & (df_cleaned.Timestamp < "1998-01-01")]

# Min-max normalisation
df_cleaned_1995["Re_di_norm"] = (df_cleaned_1995["Re_di"]-df_cleaned_1995["Re_di"].min())/(df_cleaned_1995["Re_di"].max()-df_cleaned_1995["Re_di"].min())
df_cleaned_1995["Re_lt_norm"] = (df_cleaned_1995["Re_lt"]-df_cleaned_1995["Re_lt"].min())/(df_cleaned_1995["Re_lt"].max()-df_cleaned_1995["Re_lt"].min())
df_cleaned_1995["Re_tb_norm"] = (df_cleaned_1995["Re_tb"]-df_cleaned_1995["Re_tb"].min())/(df_cleaned_1995["Re_tb"].max()-df_cleaned_1995["Re_tb"].min())

sns.lineplot(data=df_cleaned_1995, x="Timestamp", y="Re_di_norm", label="Re_di")
sns.lineplot(data=df_cleaned_1995, x="Timestamp", y="Re_lt_norm", label="Re_lt")
sns.lineplot(data=df_cleaned_1995, x="Timestamp", y="Re_tb_norm", label="Re_tb")

plt.semilogy()

plt.savefig("plots/re_time_series.png")
plt.show()