
#### ANALYSING and PLOTTING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({'font.size': 9})

df_l1_cleaned = pd.read_csv("data/processed/wind_dataset_l1_cleaned.csv")
df_l1_cleaned.Timestamp = pd.to_datetime(df_l1_cleaned.Timestamp)
df_l1_cleaned.set_index("Timestamp", inplace=True)
df_l1_cleaned.sort_index(inplace=True)
print(df_l1_cleaned.info())

### OVERLAPPING HISTOGRAMS OF CORRECTED AND UNCORRECTED TAYLOR SCALES ###

##### New method (densities, linear scale, hatching)

fig, ax = plt.subplots(figsize=(3.3,1.8), constrained_layout=True)

# Compute density estimates
density1 = sns.kdeplot(df_l1_cleaned.lambda_t_raw).get_lines()[0].get_data()
density2 = sns.kdeplot(df_l1_cleaned.lambda_t).get_lines()[1].get_data()

# Plot the densities without filling them
plt.plot(density1[0], density1[1], color="black", label="$\lambda_{T}^{extra}$")
plt.plot(density2[0], density2[1], color="green", label="$\lambda_{T}$")

# Fill between the density curve and the x-axis with hatching for each histogram
# NB: Repeat character to make hatching denser
plt.fill_between(density1[0], density1[1], color="none", hatch="///", edgecolor="black", alpha=0.5)
plt.fill_between(density2[0], density2[1], color="none", hatch="\\\\\\", edgecolor="green", alpha=0.5)

plt.axvline(df_l1_cleaned.lambda_t_raw.mean(), c="black", ls='--', alpha = 0.5)
plt.axvline(df_l1_cleaned.lambda_t.mean(), c="green", ls='--', alpha = 0.5)

plt.text(1000, 0.00024, "$\lambda_{T}$", color="green", size = 13)
plt.text(500, 0.00048, "Mean = {:.0f}".format((df_l1_cleaned.lambda_t.mean())), color="green")

plt.text(6000, 0.00024, "$\lambda_{T}^{extra}$", color="black", size = 13)
plt.text(4800, 0.00048, "Mean = {:.0f}".format((df_l1_cleaned.lambda_t_raw.mean())), color="black")

#plt.legend(loc="upper right")
plt.xlim(0,9000)
plt.ylim(0,0.0006)
plt.ylabel("Density")
plt.yticks([])
plt.xlabel("Taylor scale (km)")

plt.savefig("plots/final/taylor_overlapping_hist_v2.pdf")
plt.show()



def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    x = pd.Series(x)
    y = pd.Series(y)
    rp = x.corr(y, "pearson")
    rs = x.corr(y, "spearman")
    ax = ax or plt.gca()
    ax.annotate(f'Pearson:\n{rp:.2f}', xy=(.95, .3), xycoords=ax.transAxes, ha="right")
    ax.annotate(f'Spearman:\n{rs:.2f}', xy=(.95, .1), xycoords=ax.transAxes, ha="right")

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

# For these plots, need to use this instead of constrained_layout
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


#### PLOTTING ADDITIONAL BREAKSCALE RELATIONSHIPS ####

# Create side-by-side subplots with a shared y-axis

# Want to plot the following:
# - lambda d vs. lambda c x delta b ^ -1.737
# - lambda d vs. lambda T x delta b ^ -0.579

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.histplot(x=df_l1_cleaned["lambda_t"]*df_l1_cleaned["db"]**-0.579, y=df_l1_cleaned["tb"], ax=axes[0])
sns.histplot(x=df_l1_cleaned["lambda_c_fit"]*df_l1_cleaned["db"]**-1.737, y=df_l1_cleaned["tb"], ax=axes[1])

axes[0].set_ylabel("$\lambda_d$ (sec)")
axes[0].set_xlabel("$\lambda_{T} \delta b^{-0.579}$")
axes[1].set_xlabel("$\lambda_{C} \delta b^{-1.737}$")

#plt.tight_layout()
# axes[0].set_xlim(0, 2500)
axes[1].set_xlim(0, 150000)
axes[0].set_ylim(1e-1, 2)
axes[1].set_ylim(1e-1, 2)
# axes[0].tick_params(direction='in')
# axes[1].tick_params(direction='in')

plt.show()

# Correlation coefficients
x = df_l1_cleaned["lambda_c_fit"]*df_l1_cleaned["db"]**-1.737
y = df_l1_cleaned["lambda_t"]*df_l1_cleaned["db"]**-0.579
x.corr(df_l1_cleaned["tb"], "pearson")
y.corr(df_l1_cleaned["tb"], "pearson")

###################################

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


#######################################################

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