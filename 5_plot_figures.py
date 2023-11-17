
#### ANALYSING and PLOTTING WIND DATABASE ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

plt.rcParams.update({'font.size': 9})
plt.rc('text', usetex=True) # Set default font to Latex font

df_l1_cleaned = pd.read_csv("latest_results/wind_dataset_l1_cleaned.csv")
df_l1_cleaned.Timestamp = pd.to_datetime(df_l1_cleaned.Timestamp)
df_l1_cleaned.set_index("Timestamp", inplace=True)
df_l1_cleaned.sort_index(inplace=True)
print(df_l1_cleaned.info())

### OVERLAPPING HISTOGRAMS OF CORRECTED AND UNCORRECTED TAYLOR SCALES ###

# Summary statistics
df_l1_cleaned[['lambda_t_raw', 'lambda_t']].describe()

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
plt.text(800, 0.00048, "Mean = {:.0f}".format((df_l1_cleaned.lambda_t.mean())), color="green")

plt.text(6000, 0.00024, "$\lambda_{T}^\\mathrm{ext}$", color="black", size = 13)
plt.text(5000, 0.00048, "Mean = {:.0f}".format((df_l1_cleaned.lambda_t_raw.mean())), color="black")

#plt.legend(loc="upper right")
plt.xlim(0,9000)
plt.ylim(0,0.0006)
plt.ylabel("Density")
plt.yticks([])
plt.xlabel("Taylor scale (km)")

plt.savefig("plots/final/taylor_scale_hist.pdf")
plt.show()

def corrfunc(x, y, ax=None, loc=[.95, .25], **kws):
    """Plot the Pearson and Spearman correlation coefficients on a plot."""
    x = pd.Series(x)
    y = pd.Series(y)
    rp = x.corr(y, "pearson")
    rs = x.corr(y, "spearman")
    ax = ax or plt.gca()
    ax.annotate(f'Pearson:\n{rp:.2f}', xy=(loc[0],loc[1]), xycoords=ax.transAxes, ha="right")
    ax.annotate(f'Spearman:\n{rs:.2f}', xy=(loc[0],loc[1]-0.2), xycoords=ax.transAxes, ha="right")

def plot_regression(ax, x, y, fit_type='linear', color='red', loc=[0.05, 0.9]):
    
    if fit_type == 'log-log':
        # Take the logarithm of the data for power-law fit
        x_data = np.log(x)
        y_data = np.log(y)
    elif fit_type == 'linear':
        x_data = x
        y_data = y
    else:
        raise ValueError("fit_type must be either 'linear' or 'log-log'")
    
    x_data = sm.add_constant(x_data)
    model = sm.OLS(y_data, x_data, missing='drop')
    results = model.fit()

    intercept, slope = results.params

    # Determine the format string based on the sign of the intercept
    if intercept < 0:
        intercept_format = f"{intercept:.2f}"
    else:
        intercept_format = f"+{intercept:.2f}"

    # Add the regression line to the plot based on fit_type
    if fit_type == 'log-log':
         # Plot the power-law relationship in the original space
        ax.plot(x, (x**slope)*np.exp(intercept), color=color)
        
        # Add the regression equation to the plot
        equation_format = f"ln(y) = {slope:.2f}ln(x){intercept_format}"
        ax.text(loc[0], loc[1], equation_format, transform=ax.transAxes, color=color, bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', boxstyle='round,pad=1'))
        
    elif fit_type == 'linear':
        # Plot the linear relationship in the original space
        ax.plot(x, x*slope + intercept, color=color)
        
        # Add the regression equation to the plot
        equation_format = f"y = {slope:.2f}x{intercept_format}"
        ax.text(loc[0], loc[1], equation_format, transform=ax.transAxes, color=color)

# Summary statistics for table
df_l1_cleaned[['Re_tb', 'Re_di', 'Re_lt']].describe()
df_l1_cleaned[['Re_tb', 'Re_di', 'Re_lt']].sem() # standard errors

# Correcting Re values with pre-factors 
# SHOULD BE ABLE TO DELETE NOW AS DONE IN ANALYTICAL_VARS SCRIPT

# df_l1_cleaned["Re_tb"] = df_l1_cleaned["Re_tb"]*3
# df_l1_cleaned["Re_di"] = df_l1_cleaned["Re_di"]*3
# df_l1_cleaned["Re_lt"] = df_l1_cleaned["Re_lt"]*50

# Subsetting data to highlight main density of points
df_subset = df_l1_cleaned[df_l1_cleaned.Re_lt/df_l1_cleaned.Re_tb < 50]
df_not_subset = df_l1_cleaned[~df_l1_cleaned.index.isin(df_subset.index)]

# Alt grouping, keeping everything in one dataset
df_l1_cleaned['Group'] = 'outliers'
df_l1_cleaned.loc[df_subset.index, 'Group'] = 'cleaned'

# What prop are in this new subset?
len(df_subset)/len(df_l1_cleaned)

# What are the Re correlations now?
df_subset[['Re_tb', 'Re_di', 'Re_lt']].corr(method='pearson')

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

# Plot the marginal histograms
sns.kdeplot(data=df_l1_cleaned, x="Re_tb", ax=ax_marg_x_0, log_scale=True, color="black")
sns.kdeplot(data=df_l1_cleaned, x="Re_di", ax=ax_marg_x_1, log_scale=True, color="black")
sns.kdeplot(data=df_l1_cleaned, x="Re_lt", ax=ax_marg_x_2, log_scale=True, color="black")


# Plot the points NOT in the subset (e.g., in light gray)
# sns.histplot(ax=ax_joint_0, data=df_not_subset, x="Re_tb", y="Re_lt", log_scale=True, color="lightgray", bins=(bin_edges_x_tb, bin_edges_y_lt))
# sns.histplot(ax=ax_joint_1, data=df_not_subset, x="Re_di", y="Re_tb", log_scale=True, color="lightgray", bins=(bin_edges_x_di, bin_edges_x_tb))
# sns.histplot(ax=ax_joint_2, data=df_not_subset, x="Re_lt", y="Re_di", log_scale=True, color="lightgray", bins=(bin_edges_y_lt, bin_edges_x_di))

# Overlay the points in the subset (e.g., in red)
# sns.histplot(ax=ax_joint_0, data=df_subset, x="Re_tb", y="Re_lt", log_scale=True, color="red")
# sns.histplot(ax=ax_joint_1, data=df_subset, x="Re_di", y="Re_tb", log_scale=True, color="red")
# sns.histplot(ax=ax_joint_2, data=df_subset, x="Re_lt", y="Re_di", log_scale=True, color="red")

# Define the colors for the groups
current_palette = sns.color_palette()

# Assign the first two colors to your groups
colors = {
    "cleaned": current_palette[0],
    "outliers": "lightgray"
}

# Plot the joint histograms

sns.histplot(
    ax = ax_joint_0, 
    data=df_l1_cleaned, 
    x="Re_tb", 
    y="Re_lt", 
    log_scale=True, 
    hue='Group', 
    hue_order=["outliers", "cleaned"],
    palette=colors)

corrfunc(df_l1_cleaned["Re_tb"], df_l1_cleaned["Re_lt"], ax_joint_0, loc=(.95, .4))
ax_joint_0.set_xlabel("$Re_{t_b}$")
ax_joint_0.set_ylabel("$Re_{\lambda_T}$")

sns.histplot(ax = ax_joint_1, data=df_l1_cleaned, x="Re_di", y="Re_tb", log_scale=True, hue='Group', hue_order=["outliers", "cleaned"], palette=colors)
corrfunc(df_l1_cleaned["Re_di"], df_l1_cleaned["Re_tb"], ax_joint_1)
ax_joint_1.set_xlabel("$Re_{d_i}$")
ax_joint_1.set_ylabel("$Re_{t_b}$")

sns.histplot(ax = ax_joint_2, data=df_l1_cleaned, x="Re_lt", y="Re_di", log_scale=True, hue='Group', hue_order=["outliers", "cleaned"], palette=colors)
corrfunc(df_l1_cleaned["Re_lt"], df_l1_cleaned["Re_di"], ax_joint_2)
ax_joint_2.set_xlabel("$Re_{\lambda_T}$")
ax_joint_2.set_ylabel("$Re_{d_i}$")

for ax in [ax_marg_x_0, ax_marg_x_1, ax_marg_x_2]:
    ax.set_ylim(0, 1.2)
    ax.set_xlim(1e3, 1e8)
    ax.axis('off')

for ax in [ax_joint_0, ax_joint_1, ax_joint_2]:
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(1e3, 1e8)
    ax.plot([1e3, 1e8], [1e3, 1e8], linestyle='--', linewidth=1.0, c = "black")
    ax.plot([1e3, 1e8], [1e3, 1e8], linestyle='--', linewidth=1.0, c = "black")
    ax.plot([1e3, 1e8], [1e3, 1e8], linestyle='--', linewidth=1.0, c = "black")
    ax.set_xticks([1e4,1e6,1e8])
    ax.set_yticks([1e4,1e6,1e8])
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.minorticks_off()
    ax.grid()

ax_joint_0.tick_params(direction='in')
ax_joint_1.tick_params(direction='in', labelleft=False)
ax_joint_2.tick_params(direction='in', labelleft=False)

# Apply the regression function OF THE CLEANED SUBSETS to each of the joint plots
plot_regression(ax_joint_0, df_subset["Re_tb"], df_subset["Re_lt"], fit_type='log-log', color="darkblue", loc=[0.2, 0.05])
plot_regression(ax_joint_1, df_subset["Re_di"], df_subset["Re_tb"], fit_type='log-log', color="darkblue")  
plot_regression(ax_joint_2, df_subset["Re_lt"], df_subset["Re_di"], fit_type='log-log', color="darkblue")

# For these plots, need to use this instead of constrained_layout
fig.tight_layout()

# Remove the legends (labelling the groups)
ax_joint_0.get_legend().remove()
ax_joint_1.get_legend().remove()
ax_joint_2.get_legend().remove()

# Save/show the plot
plt.savefig("plots/final/reynolds_hist.pdf")
plt.show()


# NOW FOR CORR SCALES

# Summary statistics for table
df_l1_cleaned[['lambda_c_fit', 'lambda_c_e', 'lambda_c_int']].describe()
df_l1_cleaned[['lambda_c_fit', 'lambda_c_e', 'lambda_c_int']].sem() # standard errors

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
ax_joint_0.set_xlabel("$\lambda_{C}^\\mathrm{fit}$ (km)")
ax_joint_0.set_ylabel("$\lambda_{C}^{1/e}$ (km)")

sns.histplot(ax = ax_joint_1, data=df_l1_cleaned, x="lambda_c_e", y="lambda_c_int", log_scale=True)
corrfunc(df_l1_cleaned["lambda_c_e"], df_l1_cleaned["lambda_c_int"], ax_joint_1)
ax_joint_1.set_xlabel("$\lambda_{C}^{1/e}$ (km)")
ax_joint_1.set_ylabel("$\lambda_{C}^\\mathrm{int}$ (km)")

sns.histplot(ax = ax_joint_2, data=df_l1_cleaned, x="lambda_c_int", y="lambda_c_fit", log_scale=True)
corrfunc(df_l1_cleaned["lambda_c_int"], df_l1_cleaned["lambda_c_fit"], ax_joint_2)
ax_joint_2.set_xlabel("$\lambda_{C}^\\mathrm{int}$ (km)")
ax_joint_2.set_ylabel("$\lambda_{C}^\\mathrm{fit}$ (km)")

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
plt.savefig("plots/final/corr_scale_hist.pdf")
plt.show()


#### PLOTTING ADDITIONAL BREAKSCALE RELATIONSHIPS ####

# Create side-by-side subplots with a shared y-axis

# Want to plot the following:
# - lambda d vs. lambda c x delta b ^ -1.737
# - lambda d vs. lambda T x delta b ^ -0.579

# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# sns.histplot(x=df_l1_cleaned["lambda_t"]*df_l1_cleaned["db"]**-0.579, y=df_l1_cleaned["tb"], ax=axes[0])
# sns.histplot(x=df_l1_cleaned["lambda_c_fit"]*df_l1_cleaned["db"]**-1.737, y=df_l1_cleaned["tb"], ax=axes[1])

# axes[0].set_ylabel("$\lambda_d$ (sec)")
# axes[0].set_xlabel("$\lambda_{T} \delta b^{-0.579}$")
# axes[1].set_xlabel("$\lambda_{C} \delta b^{-1.737}$")

# #plt.tight_layout()
# # axes[0].set_xlim(0, 2500)
# axes[1].set_xlim(0, 150000)
# axes[0].set_ylim(1e-1, 2)
# axes[1].set_ylim(1e-1, 2)
# # axes[0].tick_params(direction='in')
# # axes[1].tick_params(direction='in')

# plt.show()

# # Correlation coefficients
# x = df_l1_cleaned["lambda_c_fit"]*df_l1_cleaned["db"]**-1.737
# y = df_l1_cleaned["lambda_t"]*df_l1_cleaned["db"]**-0.579
# x.corr(df_l1_cleaned["tb"], "pearson")
# y.corr(df_l1_cleaned["tb"], "pearson")

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