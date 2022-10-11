import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/db_wind.csv")

# Reproducing the plot for PSP data from Phillips et al.

fig, ax = plt.subplots()

ax.errorbar(np.arange(len(df)), df.ttc, yerr=df.ttc_std, fmt='o', linewidth=2, capsize=6)
ax.set(xlim=(-2, 15), ylim=(0, 15))
plt.axhline(df.ttc.mean())
plt.ylabel("Taylor microscale ($\\lambda_T$) [km]")
plt.xlabel("Interval Number")
plt.show()

# See R script for correlation plots