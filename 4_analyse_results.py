#### ANALYSING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
#df.loc[df["Re_lt"] > 1e7, "Re_lt"] = np.nan
#df.loc[df["Re_di"] > 1e7, "Re_lt"] = np.nan

df[["di", "vsw", "ttu", "Re_di", "Re_lt"]].describe().round()
# CHECK MAX VALS

df.to_csv("data/processed/db_wind_with_re.csv")

######################################################

df = pd.read_csv("data/processed/db_wind_with_re.csv")

# Drop points where ratio > 100 and ratio < 0.01 
# Save these extreme cases as case studies (as with strange slopes), but exclude from main statistical analysis


### Scatterplots and histograms of Re ###
# Add equality line as well as regression line

#g = sns.jointplot(data=df, x="Re_di", y="Re_lt", kind="hex", color="#4CB391")
g = sns.JointGrid(data=df, x="Re_di", y="Re_lt")

g.ax_joint.set(xscale = "log", yscale="log")
g.plot_joint(sns.regplot, scatter=False)

# Create an inset legend for the histogram colorbar
#cax = g.figure.add_axes([.15, .55, .02, .2])

# Add the joint and marginal histogram plots
g.plot_joint(
    sns.histplot
    #, discrete=(True, False),
#    cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax
)
g.plot_marginals(sns.histplot
#, element="step", color="#03012d"
)

plt.show()

### Time series of Re ###
# Min-max normalise

df.Timestamp = pd.to_datetime(df.Timestamp)

sns.lineplot(data=df[:2000], x="Timestamp", y="Re_lt", label="Re_lt")
sns.lineplot(data=df[:2000], x="Timestamp", y="Re_di", label = "Re_di")
plt.semilogy()
plt.show()