import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("data/processed/db_wind.csv")

df["Re_lt"] = (df["tcf"]/df["ttc"])**2
df["Re_di"] = ((df["tcf"]*df["vsw"])/df["di"])**(4/3)

df["Re_lt"][df["ttc"]==df["ttc"].min()] = np.nan # 2016-01-25 12:00:00 - CHECK THIS
df.to_csv("data/processed/db_wind_with_re.csv")

df[["Re_di", "Re_lt"]].describe().round(-3)

fig, axs = plt.subplots(ncols=2)
axs[0].hist(df["Re_di"])
axs[0].hist(df["Re_lt"])
axs[0].legend(['Re_di', 'Re_lt'])
axs[1].scatter(df["Re_di"], df["Re_lt"], c="black")
axs[1].set_xlabel("Re_di")
axs[1].set_ylabel("Re_lt")
plt.show()

