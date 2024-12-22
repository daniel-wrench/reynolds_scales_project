## Spectrum comparison for intervals with different energies
# This is the update to the spectrum "cartoon" which demonstrates the important of the Taylor scale Re in capturing the different energies (fluctuation amplitudes)

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from sunpy.util import SunpyUserWarning

# Suppress SunpyUserWarning warnings (referring to missing metadata for unwanted columns)
warnings.simplefilter("ignore", SunpyUserWarning)

import src.params as params
import src.utils as utils

# Set fontsize of plots (check matches utils.py)
plt.rcParams.update({"font.size": 9})
plt.rc("text", usetex=True)

mag_vars = ["BF1", "BGSE_0", "BGSE_1", "BGSE_2"]

# Reading in dataframe to get values for qi, di, etc.
df_wind = pd.read_csv("wind_dataset.csv", index_col="Timestamp")
df_wind.index = pd.to_datetime(df_wind.index)

# Min and max of db. Want to compute the intervals of these.
timestamp_1 = "1996-10-15 00:00:00"
timestamp_2 = "2004-02-28 12:00"

df_wind.loc[[timestamp_1, timestamp_2]][
    ["tcf", "dp", "db", "qi", "qk", "tb", "Re_di", "Re_tb", "Re_lt", "ttc"]
]


def process_get_spectra(file, timestamp):
    print(file, timestamp)

    data_raw = TimeSeries(file, concatenate=True, allow_errors=False)

    df_raw = data_raw.to_dataframe()

    df_raw = df_raw.loc[:, mag_vars]

    df_raw = df_raw.rename(
        columns={
            mag_vars[0]: "Bwind",
            mag_vars[1]: "Bx",
            mag_vars[2]: "By",
            mag_vars[3]: "Bz",
        }
    )

    df_hr = df_raw.resample(params.dt_hr).mean()

    int_hr = df_hr[
        pd.to_datetime(timestamp) : pd.to_datetime(timestamp)
        + pd.to_timedelta(params.int_size)
    ]
    int_hr.shape

    missing = int_hr.iloc[:, 0].isna().sum() / len(int_hr)
    print(missing)

    if missing > 0.4:
        # Replacing values in lists with na
        print("Large missing %")
    else:
        int_hr = int_hr.interpolate().ffill().bfill()

    (z_i, z_k, spectral_break, f_periodogram, p_smooth, fig, ax) = (
        utils.compute_spectral_stats(
            [int_hr.Bx, int_hr.By, int_hr.Bz],
            f_min_inertial=params.f_min_inertial,
            f_max_inertial=params.f_max_inertial,
            f_min_kinetic=params.f_min_kinetic,
            f_max_kinetic=params.f_max_kinetic,
            plot=True,
        )
    )

    return z_i, z_k, spectral_break, f_periodogram, p_smooth, fig, ax


trange = a.Time("2004-02-28 00:00", "2004-02-29 00:00")
dataset = a.cdaweb.Dataset("WI_H2_MFI")
result = Fido.search(trange, dataset)
print(result)

downloaded_files = Fido.fetch(result[0, 0], path="data/spare_data/")
print(downloaded_files)

z_i1, z_k1, spectral_break1, f_periodogram1, p_smooth1, fig, ax = process_get_spectra(
    "data/spare_data/wi_h2_mfi_19961015_v05.cdf", timestamp_1
)
plt.show()

z_i2, z_k2, spectral_break2, f_periodogram2, p_smooth2, fig, ax = process_get_spectra(
    "data/spare_data/wi_h2_mfi_20040228_v05.cdf", timestamp_2
)
plt.show()

df_wind["f_cf"] = 1 / (df_wind["tcf"] * 2 * np.pi)
df_wind["f_di"] = df_wind["v_r"] / (df_wind["dp"] * 2 * np.pi)

fig, ax = plt.subplots(figsize=(3.3, 2))
ax.set_ylim(1e-6, 1e8)
ax.set_xlim(1e-5, 5)

ax.plot(f_periodogram1, p_smooth1, color="black")
ax.plot(f_periodogram2, p_smooth2, color="gray")
ax.semilogx()
ax.semilogy()
ax.set_xlabel("$\log(k)$")
ax.set_ylabel("$\log(E(k))$")
ax.axvline(df_wind.loc[timestamp_1, "f_di"], color="black")
ax.axvline(df_wind.loc[timestamp_2, "f_di"], color="gray")
# ax.axvline(5e-1, label = "$\lambda_d$", color='black', alpha = 0.6)
ax.axvline(df_wind.loc[timestamp_1, "f_cf"], color="black")
ax.axvline(df_wind.loc[timestamp_2, "f_cf"], color="gray")
ax.text(2e-4, 1e-5, "$\lambda_C$")
ax.text(2e-1, 1e-5, "$d_i$")
ax.text(
    2e-3,
    1e6,
    "$Re_{{d_i}}={0:.0f}$".format(round(df_wind.loc[timestamp_2, "Re_di"], -3)),
    color="gray",
)
ax.text(
    2e-3,
    1e5,
    "$Re_{{d_i}}={0:.0f}$".format(round(df_wind.loc[timestamp_1, "Re_di"], -3)),
    color="black",
)
ax.text(
    2e-3,
    5e-3,
    "$Re_{{\lambda_T}}={0:.0f}$".format(round(df_wind.loc[timestamp_2, "Re_lt"], -3)),
    color="gray",
)
ax.text(
    2e-3,
    5e-4,
    "$Re_{{\lambda_T}}={0:.0f}$".format(round(df_wind.loc[timestamp_1, "Re_lt"], -3)),
    color="black",
)

ax.set_xticks([])
ax.set_xticks([], minor=True)
ax.set_yticks([])

plt.show()

# Looks good, but for some reason v different numbers, different di's from previous version...

# fig.savefig("plots/final/spectrum_comparison.pdf")
# fig.savefig("plots/final/spectrum_comparison.pdf")
# fig.savefig("plots/final/spectrum_comparison.pdf")
