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



print("SAVING SETTING-EVALUATION PLOTS")

# # Smallest tcf
# utils.compute_outer_scale_exp_fit(
#     time_lags=time_lags_lr,
#     acf=acf_lr_list[df_complete.index[df_complete["tcf"]
#                                       == df_complete["tcf"].min()][0]],
#     seconds_to_fit=np.round(
#         2*df_complete.loc[df_complete["tcf"] == df_complete["tcf"].min(), "tce"]),
#     save=True,
#     figname="tcf_smallest")

# # ~ Median tcf
# median_ish = df_complete.sort_values("tcf").reset_index()[
#     "tcf"][round(len(acf_hr_list)/2)]  # Fix index

# utils.compute_outer_scale_exp_fit(
#     time_lags=time_lags_lr,
#     acf=acf_lr_list[df_complete.index[df_complete["tcf"] == median_ish][0]],
#     seconds_to_fit=np.round(
#         2*df_complete.loc[df_complete["tcf"] == median_ish, "tce"]),
#     save=True,
#     figname="tcf_median")

# # Largest tcf
# utils.compute_outer_scale_exp_fit(
#     time_lags=time_lags_lr,
#     acf=acf_lr_list[df_complete.index[df_complete["tcf"]
#                                       == df_complete["tcf"].max()][0]],
#     seconds_to_fit=np.round(
#         2*df_complete.loc[df_complete["tcf"] == df_complete["tcf"].max(), "tce"]),
#     save=True,
#     figname="tcf_largest")

# # Smallest ttc acf
# plt.plot(time_lags_hr,
#          acf_hr_list[df_complete.index[df_complete["ttc"] == df_complete["ttc"].min()][0]])
# plt.title("ttc_smallest_acf")
# plt.savefig("data/processed/ttc_smallest_acf.png", bbox_inches='tight')
# plt.close()

# # Smallest ttc fitting
# utils.compute_taylor_chuychai(
#     time_lags=time_lags_hr,
#     acf=acf_hr_list[df_complete.index[df_complete["ttc"]
#                                       == df_complete["ttc"].min()][0]],
#     tau_min=10,
#     tau_max=50,
#     q=kinetic_slope_list[df_complete.index[df_complete["ttc"]
#                                            == df_complete["ttc"].min()][0]],
#     save=True,
#     figname="ttc_smallest")

# # ~ Median ttc acf
# median_ish = df_complete.sort_values("ttc").reset_index()[
#     "ttc"][round(len(acf_hr_list)/2)]

# plt.plot(time_lags_hr,
#          acf_hr_list[df_complete.index[df_complete["ttc"] == median_ish][0]])
# plt.title("median_ttc_acf")
# plt.savefig("data/processed/ttc_median_acf.png", bbox_inches='tight')
# plt.close()

# # ~ Median ttc fitting
# utils.compute_taylor_chuychai(
#     time_lags=time_lags_hr,
#     acf=acf_hr_list[df_complete.index[df_complete["ttc"] == median_ish][0]],
#     tau_min=10,
#     tau_max=50,
#     q=kinetic_slope_list[df_complete.index[df_complete["ttc"]
#                                            == median_ish][0]],
#     save=True,
#     figname="ttc_median")

# # Largest ttc acf
# plt.plot(time_lags_hr,
#          acf_hr_list[df_complete.index[df_complete["ttc"] == df_complete["ttc"].max()][0]])
# plt.title("largest_ttc_acf")
# plt.savefig("data/processed/ttc_largest_acf.png", bbox_inches='tight')
# plt.close()

# # Largest ttc fitting
# utils.compute_taylor_chuychai(
#     time_lags=time_lags_hr,
#     acf=acf_hr_list[df_complete.index[df_complete["ttc"]
#                                       == df_complete["ttc"].max()][0]],
#     tau_min=10,
#     tau_max=50,
#     q=kinetic_slope_list[df_complete.index[df_complete["ttc"]
#                                            == df_complete["ttc"].max()][0]],
#     save=True,
#     figname="ttc_largest")
