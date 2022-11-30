#### ANALYSING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/processed/db_wind_95_98.csv")
# df_2 = pd.read_csv("data/processed/db_wind_99_07.csv")

# df_1 = df_1.set_index("Timestamp").sort_index()
# df_2 = df_2.set_index("Timestamp").sort_index()
# df_merged = pd.concat([df_1, df_2], verify_integrity=True)
# df_merged.index = pd.to_datetime(df_merged.index)

# df_merged["1998-12-30":"1999-01-03"]

# BEFORE DOING THE FOLLOWING, EXCLUDE OMNI VARS FROM MERGE AS THESE ARE DOUBLING UP, SUM WILL BE BAD
# Dealing with some occasional duplication of timestamps due to a timestamp at the end of a file
# also appearing at the start of the next file 
#df_merged = df_merged.groupby(df_merged.index).agg(sum)
# Dealing with any resultant 0s from summing to NAs together
#df_merged = df_merged.replace(0, np.nan)

# df_merged_cleaned = df_merged[~df_merged.index.duplicated(keep='last')]

# df_merged_cleaned["1998-12-30":"1999-01-03"]

# Calculating Reynolds number with three different methods

df["Re_lt"] = (df["tcf"]/df["ttc"])**2
df["Re_di"] = ((df["tcf"]*df["vsw"])/df["di"])**(4/3)
df["Re_tb"] = ((df["tcf"])/(1/df["tb"]))**(4/3)

df[["tcf", "ttc", "Re_di", "Re_lt", "Re_tb"]].describe()
np.mean(df.Re_lt).round(-4)
np.mean(df.Re_di).round(-4)
np.mean(df.Re_tb).round(-4)

# Data cleaning
df.loc[df["Re_lt"] > 1e7, "Re_lt"] = np.nan
#df.loc[df["Re_di"] > 1e7, "Re_lt"] = np.nan

df[["di", "vsw", "ttu", "Re_di", "Re_lt"]].describe().round()
# CHECK MAX VALS

df.to_csv("data/processed/db_wind_with_re.csv", index=False)

######################################################

df = pd.read_csv("data/processed/db_wind_with_re.csv")

# Drop points where ratio > 100 and ratio < 0.01 
# Save these extreme cases as case studies (as with strange slopes), but exclude from main statistical analysis


### Scatterplots and histograms of Re ###

g = sns.JointGrid(
    data=df, 
    x="Re_di", 
    y="Re_lt", 
    xlim = (1e4, 3e6), 
    ylim=(1e3,1e7)
    )

g.ax_joint.set(xscale = "log", yscale="log")

# Create an inset legend for the histogram colorbar
#cax = g.figure.add_axes([.15, .55, .02, .2])

# Add the joint and marginal histogram plots
g.plot_joint(
    sns.histplot #,
    #cmap="light:#03012d", 
    #pmax=.8, 
    #cbar=True
)
g.plot_marginals(
    sns.histplot #, or kdeplot if you want densities 
    #element="step", 
    #color="#03012d"
)

# Draw a line of x=y 
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, '-r')

# Draw a regression line
# (for some reason the following looks weird)
#g.plot_joint(sns.regplot, scatter=False, ci=False)

#fit = np.polyfit(np.array(df_clean["Re_di"]), np.array(df_clean["Re_lt"]), deg=1)
#g.ax_joint.plot(df_clean[["Re_di"]], df_clean["Re_di"]*fit[0]+fit[1], '-b')

# df_clean = df.dropna()
# reg = LinearRegression().fit(df_clean[["Re_di"]], df_clean[["Re_lt"]])
# Re_lt_predict = reg.predict(df_clean[["Re_di"]])

plt.show()

# The coefficients
#print("Coefficients: \n", reg.coef_)
#print(reg.intercept_)

#from sklearn.metrics import mean_squared_error, r2_score
#print("Mean squared error: %.2f" % mean_squared_error(df_clean[["Re_lt"]], Re_lt_predict))
# The coefficient of determination: 1 is perfect prediction
#print("Coefficient of determination: %.2f" % r2_score(df_clean[["Re_lt"]], Re_lt_predict))




### Same but for all 3 Re methods

df_re = df[["Re_di", "Re_tb", "Re_lt"]]
df_re = df_re.dropna()
f = sns.PairGrid(df_re, diag_sharey=False, corner=False)
f.map_lower(sns.histplot, log_scale=True)
#f.map_lower(sns.regplot, scatter=False)
f.map_diag(sns.kdeplot)
plt.show()

### Time series of Re ###

df.Timestamp = pd.to_datetime(df.Timestamp)
df_1995 = df[(df.Timestamp > "1997-01-01") & (df.Timestamp < "1998-01-01")]

# Min-max normalisation
df_1995["Re_di_norm"] = (df_1995["Re_di"]-df_1995["Re_di"].min())/(df_1995["Re_di"].max()-df_1995["Re_di"].min())
df_1995["Re_lt_norm"] = (df_1995["Re_lt"]-df_1995["Re_lt"].min())/(df_1995["Re_lt"].max()-df_1995["Re_lt"].min())
df_1995["Re_tb_norm"] = (df_1995["Re_tb"]-df_1995["Re_tb"].min())/(df_1995["Re_tb"].max()-df_1995["Re_tb"].min())

sns.lineplot(data=df_1995, x="Timestamp", y="Re_di_norm", label="Re_di")
sns.lineplot(data=df_1995, x="Timestamp", y="Re_lt_norm", label="Re_lt")
sns.lineplot(data=df_1995, x="Timestamp", y="Re_tb_norm", label="Re_tb")

plt.semilogy()
plt.show()