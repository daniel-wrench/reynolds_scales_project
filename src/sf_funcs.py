import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# plt.rc("text", usetex=True)
# plt.rc("font", family="serif", serif="Computer Modern", size=16)

# Because Rāpoi can't handle latex apparently
# plt.rcParams.update(
#     {
#         "text.usetex": False,
#         "mathtext.fontset": "stix",  # Set the font to use for math
#         "font.family": "serif",  # Set the default font family
#         "font.size": 11,
#     }
# )

# Set seed for reproducibility
np.random.seed(123)


def compute_sf(
    data,
    lags,
    powers=[2],
    retain_increments=False,
    alt_estimators=True,
    pwrl_range=None,
):
    """
    Routine to compute the increments of a time series and then the mean (structure function) and standard deviation
    of the PDF of these increments, raised to the specified powers.
    Input:
            data: pd.DataFrame of data to be analysed. Must have shape (1, N) or (3, N)
            lags: The array consisting of lags, being the number of points to shift each of the series
            powers: The array consisting of the Structure orders to perform on the series for all lags
    Output:
            df: The DataFrame containing  corresponding to lags for each order in powers
    """
    # run through lags and powers so that for each power, run through each lag
    df = {}

    if data.shape[1] == 1:
        ax = data.iloc[:, 0].copy()
        for i in powers:
            array = []
            mean_array = []
            mapd_array = []
            std_array = []
            N_array = []
            for lag in lags:
                lag = int(lag)
                dax = ax.shift(-lag) - ax
                strct = dax.pow(i)

                array += [strct.values]
                strct_mean = strct.mean()
                if dax.isnull().sum() != len(dax):
                    # Otherwise this func will raise an error
                    median_abs_diff = np.nanmedian(dax)
                else:
                    median_abs_diff = np.nan
                mean_array += [strct_mean]
                mapd_array += [median_abs_diff]
                strct_std = strct.std()
                std_array += [strct_std]

                N = dax.notnull().sum()
                N_array += [N]

                df["lag"] = lags
                df["n"] = N_array
                df["sf_" + str(i)] = mean_array
                df["sf_" + str(i) + "_se"] = np.array(std_array) / np.sqrt(N_array)
                if retain_increments is True:
                    df["diffs_" + str(i)] = array
                    df["diffs_" + str(i) + "_sd"] = std_array

    elif data.shape[1] == 3:
        ax = data.iloc[:, 0].copy()
        ay = data.iloc[:, 1].copy()
        az = data.iloc[:, 2].copy()
        for i in powers:
            array = []
            mean_array = []
            mapd_array = []
            std_array = []
            N_array = []
            for lag in lags:
                lag = int(lag)
                dax = np.abs(ax.shift(-lag) - ax)
                day = np.abs(ay.shift(-lag) - ay)
                daz = np.abs(az.shift(-lag) - az)
                strct = (dax**2 + day**2 + daz**2).pow(0.5).pow(i)

                array += [strct.values]
                strct_mean = strct.mean()
                if strct.isnull().sum() != len(strct):
                    # Otherwise this func will raise an error
                    median_abs_diff = np.nanmedian(strct)
                else:
                    median_abs_diff = np.nan
                mean_array += [strct_mean]
                mapd_array += [median_abs_diff]
                strct_std = strct.std()
                std_array += [strct_std]

                N = dax.notnull().sum()
                N_array += [N]

                df["lag"] = lags
                df["n"] = N_array
                df["sf_" + str(i)] = mean_array
                df["sf_" + str(i) + "_se"] = np.array(std_array) / np.sqrt(N_array)
                if retain_increments is True:
                    df["diffs_" + str(i)] = array
                    df["diffs_" + str(i) + "_sd"] = std_array

    else:
        raise ValueError("Data is not in the shape (1, N) or (3, N)")

    df = pd.DataFrame(df, index=lags)
    if alt_estimators is True:
        df["mapd"] = mapd_array
        df["sf_2_ch"] = df["sf_0.5"] ** 4 / (0.457 + (0.494 / df["n"]))
        df["sf_2_dowd"] = (df["mapd"] ** 2) * 2.198
    # calculate sample size as a proportion of the maximum sample size (for that lag)
    df.insert(2, "missing_percent", 100 * (1 - (df["n"] / (len(ax) - df.index))))

    if pwrl_range is not None:
        # Fit a line to the log-log plot of the structure function over the given range
        min, max = pwrl_range[0], pwrl_range[1]

        slope = np.polyfit(
            np.log(df.loc[min:max, "lag"]),
            np.log(df.loc[min:max, "sf_2"]),
            1,
        )[0]

        return df, slope

    else:
        return df


def get_lag_vals_list(df, value_name="sq_diffs"):
    lag_vals_wide = pd.DataFrame(df[value_name].tolist(), index=df.index)
    lag_vals_wide.reset_index(inplace=True)  # Make the index a column
    lag_vals_wide.rename(columns={"index": "lag"}, inplace=True)
    lag_vals = pd.melt(
        lag_vals_wide, id_vars=["lag"], var_name="index", value_name=value_name
    )
    return lag_vals


def plot_sample(
    good_input,
    good_output,
    other_inputs,
    other_outputs,
    input_ind=0,
    input_versions=3,  # Either number of versions to plot or a list of versions to plot
    linear=True,
    estimator_list=["sf_2"],  # sf_2, ch, dowd
    title="SF estimation subject to missing data",
):
    if linear is False:
        ncols = 3
    else:
        ncols = 4

    # Check if n is an integer
    if not isinstance(input_versions, int):
        n = len(input_versions)
        fig, ax = plt.subplots(
            n,
            ncols,
            figsize=(ncols * 5, n * 3),
            sharex="col",
            gridspec_kw={"hspace": 0.2},
        )
        other_inputs_plot = [other_inputs[input_ind][i] for i in input_versions]
        other_outputs_plot = [other_outputs[input_ind][i] for i in input_versions]
    else:
        n = input_versions
        fig, ax = plt.subplots(
            n,
            ncols,
            figsize=(ncols * 5, n * 3),
            sharex="col",
            gridspec_kw={"hspace": 0.2},
        )
        # Before plotting, sort the n bad inputs by missing proportion
        other_inputs_plot = other_inputs[input_ind][:n]
        other_outputs_plot = other_outputs[input_ind][:n]

    sparsities = [df["missing_percent_overall"].values[0] for df in other_outputs_plot]

    sorted_lists = zip(*sorted(zip(sparsities, other_inputs_plot)))
    sparsities_ordered, other_inputs_plot = sorted_lists

    sorted_lists = zip(*sorted(zip(sparsities, other_outputs_plot)))
    sparsities_ordered, other_outputs_plot = sorted_lists

    ax[0, 0].set_title("Interpolated time series")
    ax[0, ncols - 2].set_title("$S_2(\\tau)$")
    ax[0, ncols - 1].set_title("Estimation error")

    for i in range(n):
        missing = other_outputs_plot[i]["missing_percent_overall"].values[0]
        # missing = np.isnan(ts_plot).sum() / len(ts_plot)
        ax[i, 0].plot(good_input[input_ind].values, color="grey", lw=0.8)
        ax[i, 0].plot(other_inputs_plot[i], color="black", lw=0.8)

        # Add the missing % as an annotation in the top left
        ax[i, 0].annotate(
            f"{missing*100:.2f}% missing",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0.05, 0.9),
            textcoords="axes fraction",
            transform=ax[i, 0].transAxes,
            c="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

        colors = ["C0", "darkred", "olivedrab"]
        pos_y = [0.9, 0.8, 0.7]
        for est_ind, estimator in enumerate(estimator_list):
            mape = other_outputs_plot[i][estimator + "_pe"].abs().mean()

            ax[i, 1].annotate(
                "MAPE = {:.2f}".format(mape),
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.05, pos_y[est_ind]),
                textcoords="axes fraction",
                transform=ax[i, 1].transAxes,
                color=colors[est_ind],
                bbox=dict(
                    facecolor="white",
                    edgecolor="white",
                    boxstyle="round",
                    # linestyle=":",
                ),
            )

            ax[i, ncols - 1].plot(
                other_outputs_plot[i]["missing_prop"] * 100,
                color="black",
                label="% pairs missing",
            )
            ax[i, ncols - 1].semilogx()
            ax[i, ncols - 1].set_ylim(0, 100)

            ax2 = ax[i, ncols - 1].twinx()
            ax2.tick_params(axis="y", colors="C0")
            ax2.plot(
                other_outputs_plot[i][estimator + "_pe"],
                color=colors[est_ind],
                label="% error",
                lw=0.8,
            )

            ax2.semilogx()
            ax2.set_ylim(-100, 100)
            ax2.axhline(0, color="C0", linestyle="--")
            if i == 0:
                ax2.annotate(
                    "% error",
                    xy=(1, 1),
                    xycoords="axes fraction",
                    xytext=(0.75, 0.9),
                    textcoords="axes fraction",
                    transform=ax[i, 0].transAxes,
                    c="C0",
                    bbox=dict(
                        facecolor="white", edgecolor="grey", boxstyle="round", alpha=0.7
                    ),
                )

            # Plot scatter plot and line plot for both log-scale and linear-scale
            for j in range(ncols - 2):
                j += 1

                ax[i, j].plot(
                    good_output[input_ind]["lag"],
                    good_output[input_ind][estimator],
                    color=colors[est_ind],
                    alpha=0.3,
                    linewidth=3.5,
                    label=estimator,
                )
                ax[i, j].plot(
                    other_outputs_plot[i]["lag"],
                    other_outputs_plot[i][estimator],
                    color=colors[est_ind],
                    linewidth=0.8,
                    # ls=":",
                    # label=f": {estimator}",
                )
                suffix = ""  # for the title
                # Get lag vals
                if (
                    "sq_diffs" in other_outputs_plot[i].columns
                    and len(good_input[input_ind]) < 3000
                ):
                    other_lag_vals = get_lag_vals_list(other_outputs_plot[i])
                    ax[i, j].scatter(
                        other_lag_vals["lag"],
                        other_lag_vals["sq_diffs"],
                        alpha=0.005,
                        s=1,
                    )
                    suffix = " + squared diffs"

                # Plot "confidence region" of +- x SEs
                # if estimator == "sf_2":
                #     x = 3
                #     ax[i, j].fill_between(
                #         other_outputs_plot[i]["lag"],
                #         np.maximum(
                #             other_outputs_plot[i]["sf_2"]
                #             - x * other_outputs_plot[i]["sf_2_se"],
                #             0,
                #         ),
                #         other_outputs_plot[i]["sf_2"]
                #         + x * other_outputs_plot[i]["sf_2_se"],
                #         color="C0",
                #         alpha=0.4,
                #         label=f"$\pm$ {x} SE",
                #     )

                ax[i, j].set_ylim(5e-3, 5e0)

        if linear is True:
            ax[i, 2].semilogx()
            ax[i, 2].semilogy()
        else:
            ax[i, 1].semilogx()
            ax[i, 1].semilogy()

    ax[n - 1, 0].set_xlabel("Time")
    for i in range(1, ncols):
        ax[n - 1, i].set_xlabel("Lag ($\\tau$)")
    # Remove x-axis labels for all but the bottom row
    # for i in range(n):
    #     for j in range(ncols):
    #         if i < n:
    #             ax[i, j].set_xticklabels([])

    # ax[0, ncols - 1].axhline(0, color="black", linestyle="--")
    ax[0, ncols - 1].semilogx()
    # ax[0, 1].legend(loc="lower right", frameon=True)

    ax[0, ncols - 1].annotate(
        "% pairs missing",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.9),
        textcoords="axes fraction",
        transform=ax[0, 2].transAxes,
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round", alpha=0.5),
    )

    if linear is True:
        ax[0, 1].set_title("$S_2(\\tau)$" + suffix)

    # Add overall title
    # fig.suptitle(title, size=16)

    # plt.show()


# Load in each pickle file psp_dataframes_0X.pkl and concatenate them
# into one big dataframe for each of the four dataframes


def get_all_metadata(pickle_files, include_sfs=False):
    if include_sfs is True:
        concatenated_dataframes = {
            "files_metadata": [],
            "ints_metadata": [],
            "ints_gapped_metadata": [],
            "sfs": [],
            "sfs_gapped": [],
        }

        for file in pickle_files:
            try:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    for key in concatenated_dataframes.keys():
                        concatenated_dataframes[key].append(data[key])
            except pickle.UnpicklingError:
                print(
                    f"UnpicklingError encountered in file: {file}. Skipping this file."
                )
            except EOFError:
                print(f"EOFError encountered in file: {file}. Skipping this file.")
            except Exception as e:
                print(
                    f"An unexpected error {e} occurred with file: {file}. Skipping this file."
                )

        for key in concatenated_dataframes.keys():
            if (
                key == "ints"
            ):  # Ints is a list of list of pd.Series, not a list of dataframes
                concatenated_dataframes[key] = concatenated_dataframes[key]
            else:
                concatenated_dataframes[key] = pd.concat(
                    concatenated_dataframes[key], ignore_index=True
                )

        # Access the concatenated DataFrames
        files_metadata = concatenated_dataframes["files_metadata"]
        ints_metadata = concatenated_dataframes["ints_metadata"]
        ints_gapped_metadata = concatenated_dataframes["ints_gapped_metadata"]
        sfs = concatenated_dataframes["sfs"]
        sfs_gapped = concatenated_dataframes["sfs_gapped"]

        return (
            files_metadata,
            ints_metadata,
            ints_gapped_metadata,
            sfs,
            sfs_gapped,
        )

    else:
        concatenated_dataframes = {
            "files_metadata": [],
            "ints_metadata": [],
            "ints_gapped_metadata": [],
        }

        for file in pickle_files:
            try:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    for key in concatenated_dataframes.keys():
                        concatenated_dataframes[key].append(data[key])
            except pickle.UnpicklingError:
                print(
                    f"UnpicklingError encountered in file: {file}. Skipping this file."
                )
            except EOFError:
                print(f"EOFError encountered in file: {file}. Skipping this file.")
            except Exception as e:
                print(
                    f"An unexpected error {e} occurred with file: {file}. Skipping this file."
                )

        for key in concatenated_dataframes.keys():
            if (
                key == "ints"
            ):  # Ints is a list of list of pd.Series, not a list of dataframes
                concatenated_dataframes[key] = concatenated_dataframes[key]
            else:
                concatenated_dataframes[key] = pd.concat(
                    concatenated_dataframes[key], ignore_index=True
                )

        # Access the concatenated DataFrames
        files_metadata = concatenated_dataframes["files_metadata"]
        ints_metadata = concatenated_dataframes["ints_metadata"]
        ints_gapped_metadata = concatenated_dataframes["ints_gapped_metadata"]

        return (
            files_metadata,
            ints_metadata,
            ints_gapped_metadata,
        )


def plot_error_trend_line(
    df,
    estimator="sf_2",
):
    fig, ax = plt.subplots(figsize=(5, 2), ncols=2, sharey=True)
    # plt.title(title)
    # plt.plot(lag_error_mean_i, color="black", lw=3)

    # Add second plot for scatter plot
    other_outputs_df = df[df["gap_handling"] == "naive"]

    ax[0].scatter(
        other_outputs_df["lag"],
        other_outputs_df[estimator + "_pe"],
        c=other_outputs_df["missing_percent"],
        s=0.03,
        alpha=0.4,
        cmap="plasma",
    )

    mean_error = other_outputs_df.groupby("lag")[estimator + "_pe"].mean()
    ax[0].plot(mean_error, color="royalblue", lw=3, label="Mean \% error")
    ax[0].set_title("Naive")
    ax[0].hlines(0, 1, other_outputs_df.lag.max(), color="black", linestyle="--")
    ax[0].set_xlabel("Lag ($\\tau$)")

    ax[0].annotate(
        "MAPE = {0:.2f}".format(other_outputs_df[estimator + "_pe"].abs().mean()),
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        c="black",
        size=12,
    )

    # ax[0].set_ylim(-2e2, 6e2)
    ax[0].semilogx()
    # Plot legend
    ax[0].legend(loc="lower left")
    # if y_axis_log is True:
    #     ax[0].set_yscale("symlog", linthresh=1e2)

    other_outputs_df = df[df["gap_handling"] == "lint"]
    estimator = "sf_2"

    sc = ax[1].scatter(
        other_outputs_df["lag"],
        other_outputs_df[estimator + "_pe"],
        c=other_outputs_df["missing_percent"],
        s=0.03,
        alpha=0.4,
        cmap="plasma",
    )
    mean_error = other_outputs_df.groupby("lag")[estimator + "_pe"].mean()
    ax[1].plot(mean_error, color="royalblue", lw=3, label="Mean \% error")

    ax[1].annotate(
        "MAPE = {0:.2f}".format(other_outputs_df[estimator + "_pe"].abs().mean()),
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        c="black",
        size=12,
    )

    # Change range of color bar
    ax[1].hlines(0, 1, other_outputs_df.lag.max(), color="black", linestyle="--")
    ax[1].set_ylim(-2e2, 6e2)
    ax[1].semilogx()
    # if y_axis_log is True:
    #     ax[1].set_yscale("symlog", linthresh=1e2)
    ax[1].set_title("LINT")
    ax[1].set_xlabel("Lag ($\\tau$)")
    ax[0].set_ylabel("\% error")

    cb = plt.colorbar(sc, cax=ax[1].inset_axes([1.05, 0, 0.03, 1]))
    sc.set_clim(0, 100)
    cb.set_label("\% missing")

    ax[0].set_ylim(-100, 100)
    plt.subplots_adjust(wspace=0.108)


def plot_error_trend_scatter(
    bad_outputs_df, interp_outputs_df, title="Overall % error vs. sparsity"
):
    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    sfn_mape = bad_outputs_df.groupby("missing_percent_overall")["sf_2_pe"].agg(
        lambda x: np.mean(np.abs(x))
    )

    sfn_mape_i = interp_outputs_df.groupby("missing_percent_overall")["sf_2_pe"].agg(
        lambda x: np.mean(np.abs(x))
    )
    plt.scatter(sfn_mape.index, sfn_mape.values, c="C0", label="No handling", alpha=0.5)
    plt.scatter(
        sfn_mape_i.index,
        sfn_mape_i.values,
        c="purple",
        label="Linear interp.",
        alpha=0.1,
    )

    # Add regression lines
    import statsmodels.api as sm

    x = sm.add_constant(sfn_mape.index)

    model = sm.OLS(sfn_mape.values, x)
    results = model.fit()
    plt.plot(sfn_mape.index, results.fittedvalues, c="C0")

    x_i = sm.add_constant(sfn_mape_i.index)
    model = sm.OLS(sfn_mape_i.values, x_i)
    results = model.fit()
    plt.plot(sfn_mape_i.index, results.fittedvalues, c="purple")

    plt.xlabel("Fraction of data missing overall")
    plt.ylabel("MAPE")
    plt.ylim(0, 120)
    plt.title(title)
    plt.legend(loc="upper left")
    # plt.show()


# def get_correction_lookup(
#     inputs, missing_measure, dim, comm, gap_handling="lint", n_bins=25
# ):
#     """Extract the mean error for each bin of lag and missing measure.
#     Args:
#         n_bins: The number of bins to use in each direction (x and y)
#     """

#     if dim == 2:
#         # Technically we only need rank 0 to do the following set-up,
#         # but it's easier to have all ranks do it in parallel and avoid broadcasting
#         inputs = inputs[inputs["gap_handling"] == gap_handling]

#         # Calculate the mean value in each bin
#         xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
#         yidx = np.digitize(y, yedges) - 1  # as above

#         pe_mean = np.full((n_bins, n_bins), fill_value=np.nan)
#         pe_min = np.full((n_bins, n_bins), fill_value=np.nan)
#         pe_max = np.full((n_bins, n_bins), fill_value=np.nan)
#         pe_std = np.full((n_bins, n_bins), fill_value=np.nan)
#         n = np.full((n_bins, n_bins), fill_value=np.nan)
#         scaling = np.full((n_bins, n_bins), fill_value=np.nan)
#         scaling_lower = np.full((n_bins, n_bins), fill_value=np.nan)
#         scaling_upper = np.full((n_bins, n_bins), fill_value=np.nan)

#         # Loop over every combination of bin: if there are any values
#         # in that bin, calculate the mean and standard deviation

#         for i in range(start_bin, end_bin):
#             for j in range(n_bins):
#                 if len(x[(xidx == i) & (yidx == j)]) > 0:
#                     current_bin_vals = inputs["sf_2_pe"][(xidx == i) & (yidx == j)]

#                     pe_mean[i, j] = np.nanmean(current_bin_vals)
#                     pe_std[i, j] = np.nanstd(current_bin_vals)
#                     pe_min[i, j] = np.nanmin(current_bin_vals)
#                     pe_max[i, j] = np.nanmax(current_bin_vals)
#                     n[i, j] = len(current_bin_vals)

#                     scaling[i, j] = 1 / (1 + pe_mean[i, j] / 100)
#                     scaling_lower[i, j] = 1 / (
#                         1 + (pe_mean[i, j] + 1 * pe_std[i, j]) / 100
#                     )
#                     scaling_upper[i, j] = 1 / (
#                         1 + (pe_mean[i, j] - 1 * pe_std[i, j]) / 100
#                     )

#                 else:  # If there are no values in the bin, set scaling to 1
#                     pe_mean[i, j] = np.nan
#                     pe_std[i, j] = np.nan
#                     pe_min[i, j] = np.nan
#                     pe_max[i, j] = np.nan
#                     n[i, j] = 0
#                     scaling[i, j] = 1
#                     scaling_lower[i, j] = 1
#                     scaling_upper[i, j] = 1

#         return (
#             xedges,
#             yedges,
#             pe_mean,
#             pe_std,
#             pe_min,
#             pe_max,
#             n,
#             scaling,
#             scaling_lower,
#             scaling_upper,
#         )

#     elif dim == 3:  # now we add in z variable
#         inputs = inputs[inputs["gap_handling"] == gap_handling]
#         x = inputs["lag"]
#         y = inputs[missing_measure]
#         z = inputs["sf_2"]

#         # Can use np.histogram2d to get the linear bin edges for 2D
#         xedges = (
#             np.logspace(0, np.log10(x.max()), n_bins + 1) - 0.01
#         )  # so that first lag bin starts just before 1
#         xedges[-1] = x.max() + 1
#         yedges = np.linspace(0, 100, n_bins + 1)  # Missing prop
#         zedges = np.logspace(-2, 1, n_bins + 1)  # ranges from 0.01 to 10

#         # Calculate the mean value in each bin
#         xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
#         yidx = np.digitize(y, yedges) - 1  # as above
#         zidx = np.digitize(z, zedges) - 1  # as above

#         pe_mean = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         pe_min = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         pe_max = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         pe_std = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         n = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         scaling = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         scaling_lower = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)
#         scaling_upper = np.full((n_bins, n_bins, n_bins), fill_value=np.nan)

#         for i in range(start_bin, end_bin):
#             for j in range(n_bins):
#                 for k in range(n_bins):
#                     if len(x[(xidx == i) & (yidx == j) & (zidx == k)]) > 0:
#                         current_bin_vals = inputs["sf_2_pe"][
#                             (xidx == i) & (yidx == j) & (zidx == k)
#                         ]

#                         pe_mean[i, j, k] = np.nanmean(current_bin_vals)
#                         pe_std[i, j, k] = np.nanstd(current_bin_vals)
#                         pe_min[i, j, k] = np.nanmin(current_bin_vals)
#                         pe_max[i, j, k] = np.nanmax(current_bin_vals)
#                         n[i, j, k] = len(current_bin_vals)

#                         scaling[i, j, k] = 1 / (1 + pe_mean[i, j, k] / 100)
#                         scaling_lower[i, j, k] = 1 / (
#                             1 + (pe_mean[i, j, k] + 1 * pe_std[i, j, k]) / 100
#                         )
#                         scaling_upper[i, j, k] = 1 / (
#                             1 + (pe_mean[i, j, k] - 1 * pe_std[i, j, k]) / 100
#                         )

#                     else:  # If there are no values in the bin, set scaling to 1
#                         pe_mean[i, j, k] = np.nan
#                         pe_std[i, j, k] = np.nan
#                         pe_min[i, j, k] = np.nan
#                         pe_max[i, j, k] = np.nan
#                         n[i, j, k] = 0
#                         scaling[i, j, k] = 1
#                         scaling_lower[i, j, k] = 1
#                         scaling_upper[i, j, k] = 1

#         return (
#             xedges,
#             yedges,
#             zedges,
#             pe_mean,
#             pe_std,
#             pe_min,
#             pe_max,
#             n,
#             scaling,
#             scaling_lower,
#             scaling_upper,
#         )


def plot_correction_heatmap(correction_lookup, dim, gap_handling="lint", n_bins=25):
    if dim == 2:
        xedges = correction_lookup["xedges"]
        yedges = correction_lookup["yedges"]
        pe_mean = correction_lookup["pe_mean"]

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.grid(False)
        plt.pcolormesh(
            xedges,
            yedges,
            pe_mean.T,
            cmap="bwr",
        )
        plt.grid(False)
        plt.colorbar(label="MPE")
        plt.clim(-100, 100)
        plt.xlabel("Lag ($\\tau$)")
        plt.ylabel("Missing percentage")
        plt.title(
            f"Distribution of missing proportion and lag ({gap_handling.upper()})",
            y=1.1,
        )
        ax.set_facecolor("black")
        ax.set_xscale("log")

        plt.savefig(
            f"plots/temp/train_heatmap_{n_bins}bins_{dim}d_{gap_handling.upper()}.png",
        )

    elif dim == 3:
        xedges = correction_lookup["xedges"]
        yedges = correction_lookup["yedges"]
        zedges = correction_lookup["zedges"]
        pe_mean = correction_lookup["pe_mean"]

        fig, ax = plt.subplots(1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True)
        # Remove spacing between subplots
        plt.subplots_adjust(wspace=0.2)
        plt.grid(False)
        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                xedges,
                yedges,
                pe_mean[:, :, i],
                cmap="bwr",
            )
            # plt.colorbar(label="MPE")
            c.set_clim(-100, 100)
            plt.xlabel("Lag ($\\tau$)")
            plt.ylabel("Missing proportion")
            plt.title("Distribution of missing proportion and lag")
            ax[i].set_facecolor("black")
            ax[i].semilogx()
            ax[i].set_title(f"Power bin {i+1}/{n_bins}".format(np.round(zedges[i], 2)))
            ax[i].set_xlabel("Lag ($\\tau$)")
            # Remove y-axis labels for all but the first plot
            if i > 0:
                ax[i].set_yticklabels([])
                ax[i].set_ylabel("")

        plt.savefig(
            f"plots/temp/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_power.png",
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True)
        # Remove spacing between subplots
        plt.grid(False)
        plt.subplots_adjust(wspace=0.2)
        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                yedges,
                zedges,
                pe_mean[i, :, :],
                cmap="bwr",
            )
            # plt.colorbar(label="MPE")
            c.set_clim(-100, 100)
            ax[i].set_xlabel("Missing prop")
            ax[i].set_ylabel("Power")
            plt.title("Distribution of missing proportion and lag")
            ax[i].set_facecolor("black")
            ax[i].semilogy()
            ax[i].set_title(f"Lag bin {i+1}/{n_bins}".format(np.round(zedges[i], 2)))
            ax[i].set_xlabel("Missing prop")
            # Remove y-axis labels for all but the first plot
            if i > 0:
                ax[i].set_yticklabels([])
                ax[i].set_ylabel("")

        plt.savefig(
            f"plots/temp/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_lag.png",
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(1, n_bins, figsize=(n_bins * 3, 3.5), tight_layout=True)
        # Remove spacing between subplots
        plt.grid(False)
        plt.subplots_adjust(wspace=0.2)
        for i in range(n_bins):
            ax[i].grid(False)
            c = ax[i].pcolormesh(
                xedges,
                zedges,
                pe_mean[:, i, :],
                cmap="bwr",
            )
            # plt.colorbar(label="MPE")
            c.set_clim(-100, 100)
            plt.title("Distribution of missing proportion and lag")
            ax[i].set_facecolor("black")
            ax[i].semilogx()
            ax[i].semilogy()
            ax[i].set_title(
                f"Missing prop bin {i+1}/{n_bins}".format(np.round(zedges[i], 2))
            )
            ax[i].set_xlabel("Lag ($\\tau$)")
            ax[i].set_ylabel("Power")
            # Remove y-axis labels for all but the first plot
            if i > 0:
                ax[i].set_yticklabels([])
                ax[i].set_ylabel("")

        plt.savefig(
            f"plots/temp/train_heatmap_{n_bins}bins_3d_{gap_handling.upper()}_missing.png",
            bbox_inches="tight",
        )
        plt.close()


def compute_scaling(inputs, dim, correction_lookup, n_bins=25):
    # Extract the elements of the lookup table
    if dim == 2:
        xedges = correction_lookup["xedges"]
        yedges = correction_lookup["yedges"]
        scaling = correction_lookup["scaling"]
        scaling_lower = correction_lookup["scaling_lower"]
        scaling_upper = correction_lookup["scaling_upper"]

        print(f"Loaded {dim}D lookup table with {n_bins} bins")

        # Apply the correction factor to the original data
        inputs = inputs.copy()
        inputs = inputs[inputs["gap_handling"] == "lint"]

        x = inputs["lag"]
        y = inputs["missing_percent"]

        xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
        yidx = np.digitize(y, yedges) - 1  # as above

        # Stick with original value if no bins available
        inputs["sf_2_corrected_2d"] = inputs["sf_2"].copy()
        inputs["sf_2_lower_corrected_2d"] = inputs[
            "sf_2"
        ].copy()  # Named in this slightly unwieldy way for better consistency in later wrangling
        inputs["sf_2_upper_corrected_2d"] = inputs["sf_2"].copy()

        for i in range(n_bins):
            for j in range(n_bins):
                # If there are any values, calculate the mean for that bin
                if len(x[(xidx == i) & (yidx == j)]) > 0:
                    inputs.loc[(xidx == i) & (yidx == j), "sf_2_corrected_2d"] = (
                        inputs["sf_2"][(xidx == i) & (yidx == j)] * scaling[i, j]
                    )
                    inputs.loc[(xidx == i) & (yidx == j), "sf_2_lower_corrected_2d"] = (
                        inputs["sf_2"][(xidx == i) & (yidx == j)] * scaling_lower[i, j]
                    )
                    inputs.loc[(xidx == i) & (yidx == j), "sf_2_upper_corrected_2d"] = (
                        inputs["sf_2"][(xidx == i) & (yidx == j)] * scaling_upper[i, j]
                    )

        # Smoothed version
        inputs["sf_2_corrected_2d_smoothed"] = (
            inputs["sf_2_corrected_2d"].rolling(50).mean()
        )

    elif dim == 3:
        xedges = correction_lookup["xedges"]
        yedges = correction_lookup["yedges"]
        zedges = correction_lookup["zedges"]
        scaling = correction_lookup["scaling"]
        scaling_lower = correction_lookup["scaling_lower"]
        scaling_upper = correction_lookup["scaling_upper"]

        print(f"Loaded {dim}D lookup table with {n_bins} bins")

        # Apply the correction factor to the original data
        inputs = inputs[inputs["gap_handling"] == "lint"]

        x = inputs["lag"]
        y = inputs["missing_percent"]
        z = inputs["sf_2"]

        xidx = np.digitize(x, xedges) - 1
        yidx = np.digitize(y, yedges) - 1
        zidx = np.digitize(z, zedges) - 1

        # Stick with original value if no bins available
        inputs = inputs.copy()
        inputs["sf_2_corrected_3d"] = inputs["sf_2"].copy()
        inputs["sf_2_lower_corrected_3d"] = inputs["sf_2"].copy()
        inputs["sf_2_upper_corrected_3d"] = inputs["sf_2"].copy()

        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    # If there are any values, calculate the mean for that bin
                    if len(x[(xidx == i) & (yidx == j) & (zidx == k)]) > 0:
                        inputs.loc[
                            (xidx == i) & (yidx == j) & (zidx == k), "sf_2_corrected_3d"
                        ] = (
                            inputs["sf_2"][(xidx == i) & (yidx == j) & (zidx == k)]
                            * scaling[i, j, k]
                        )
                        inputs.loc[
                            (xidx == i) & (yidx == j) & (zidx == k),
                            "sf_2_lower_corrected_3d",
                        ] = (
                            inputs["sf_2"][(xidx == i) & (yidx == j) & (zidx == k)]
                            * scaling_lower[i, j, k]
                        )
                        inputs.loc[
                            (xidx == i) & (yidx == j) & (zidx == k),
                            "sf_2_upper_corrected_3d",
                        ] = (
                            inputs["sf_2"][(xidx == i) & (yidx == j) & (zidx == k)]
                            * scaling_upper[i, j, k]
                        )

    return inputs
