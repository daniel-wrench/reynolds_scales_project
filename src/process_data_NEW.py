
import datetime
import glob
import numpy as np
import pandas as pd

import utils # add src. prefix if running interactively
import params # add src. prefix if running interactively

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

except ImportError:
    # Set default/empty single-process values if MPI is not available
    print("MPI not available, running in single-process mode.")
    class DummyComm:
        def Get_size(self):
            return 1
        def Get_rank(self):
            return 0
        def Barrier(self):
            pass
        def bcast(self, data, root=0):
            return data

    comm = DummyComm()
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = None


def get_subfolders(path):
    return sorted(glob.glob(path + "/*"))


def get_cdf_paths(subfolder):
    return sorted(glob.iglob(subfolder + "/*.cdf"))


def get_file_list(input_dir):
    file_paths = [get_cdf_paths(subfolder) for subfolder in get_subfolders(input_dir)]
    file_list = []
    for sub in file_paths:
        for cdf_file_name in sub:
            file_list.append(cdf_file_name)
    return file_list


def generate_date_strings(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    date_list = [start + datetime.timedelta(days=x) for x in range((end-start).days + 1)]
    return [date.strftime("%Y%m%d") for date in date_list]


def read_dated_file(date, file_list, varlist, newvarnames, cadence, thresholds):
    matched_files = [file for file in file_list if date in file]
    if not matched_files:
        raise ValueError(f"No files found for date {date}")
    elif len(matched_files) > 1:
        print(matched_files)
        raise ValueError(f"Multiple files found for date {date}")
    else:
        # Read in file
        try:
            df = utils.pipeline(
                matched_files[0],
                varlist=varlist,
                thresholds=thresholds,
                cadence=cadence
            )
            print("Core {0:03d} finished reading {1}: {2:.2f}% missing".format(
                rank, matched_files[0], df.iloc[:, -1].isna().sum()/len(df)*100))
            df = df.rename(columns=newvarnames)
            # print(df.head())
            return df

        except Exception as e:
            print(f"\nError reading {matched_files[0]}. Error: {e}; moving to next file")
            pass


# Initialize the file lists to be broadcasted to all cores
mfi_file_list = None
proton_file_list = None
electron_file_list = None
dates_for_cores = None

if rank == 0:
    print("#######################################")
    print("PROCESSING DATA FOR SOLAR WIND DATABASE")
    print("#######################################")

    mfi_file_list = get_file_list("data/raw/wind/mfi/mfi_h2/")
    proton_file_list = get_file_list("data/raw/wind/3dp/3dp_pm/")
    electron_file_list = get_file_list("data/raw/wind/3dp/3dp_elm2/")

    # Generate all date strings
    all_dates = generate_date_strings(params.start_date, params.end_date)

    # Split date strings among cores
    dates_for_cores = np.array_split(all_dates, size)

# Broadcast the file and date lists to all cores
mfi_file_list = comm.bcast(mfi_file_list, root=0)
proton_file_list = comm.bcast(proton_file_list, root=0)
electron_file_list = comm.bcast(electron_file_list, root=0)
dates_for_cores = comm.bcast(dates_for_cores, root=0)

# For each core, read in files for each date it has been assigned
for date in dates_for_cores[rank]:

    print("\nCORE {0:03d} READING CDF FILES FOR {1}\n".format(rank, date))

    # MFI
    mfi_df_hr = read_dated_file(date,
                    mfi_file_list,
                    [params.timestamp, params.Bwind, params.Bwind_vec],
                    {params.Bx: "Bx", params.By: "By", params.Bz: "Bz"},
                    params.dt_hr,
                    params.mag_thresh
                    )

    # PROTONS
    df_protons = read_dated_file(date,
                    proton_file_list,
                    [params.timestamp, params.np, params.nalpha, params.Tp, params.Talpha, params.V_vec],
                    {params.Vx: "Vx",
                     params.Vy: "Vy",
                     params.Vz: "Vz",
                     params.np: "np",
                     params.nalpha: "nalpha",
                     params.Tp: "Tp",
                     params.Talpha: "Talpha"},
                     params.dt_protons,
                     params.proton_thresh
                     )

    df_electrons = read_dated_file(date,
                    electron_file_list,
                    [params.timestamp, params.ne, params.Te],
                    {params.ne: "ne", params.Te: "Te"},
                    params.int_size,
                    params.electron_thresh
                    )

    # Splitting entire dataframe into a list of intervals (each themselves a dataframe)
    # First, set a start and end time for the first interval

    starttime = pd.to_datetime(date)
    endtime = starttime + pd.to_timedelta(params.int_size) - pd.to_timedelta("0.01S")
    # E.g. 2016-01-01 11:59:59.99

    # Number of intervals in the dataset
    n_int = np.round(pd.to_timedelta('24H') / pd.to_timedelta(params.int_size)).astype(int)
    # NB: If we subset timestamps that don't exist in the dataframe, they will still be included in the list, just as
    # missing dataframes. We can identify these with df.empty = True (or missing)

    # LOCAL: for plotting ACFs to check (so only run on small number of intervals locally)
    acf_hr_list = []
    velocity_acf_lr_list = []
    acf_lr_list = []

    df = pd.DataFrame({
        "Timestamp": [np.nan]*n_int,
        "missing_mfi": [np.nan]*n_int,
        "missing_3dp": [np.nan]*n_int,
        "np": [np.nan]*n_int,
        "nalpha": [np.nan]*n_int,
        "Tp": [np.nan]*n_int,
        "Talpha": [np.nan]*n_int,
        "B0": [np.nan]*n_int,
        "db": [np.nan]*n_int,
        "db_a": [np.nan]*n_int,
        "dboB0": [np.nan]*n_int,
        "V0": [np.nan]*n_int,
        "v_r": [np.nan]*n_int,
        "dv": [np.nan]*n_int,
        "zp": [np.nan]*n_int,
        "zm": [np.nan]*n_int,
        "sigma_c": [np.nan]*n_int,
        "sigma_c_abs": [np.nan]*n_int,
        "sigma_r": [np.nan]*n_int,
        "ra": [np.nan]*n_int,
        "cos_a": [np.nan]*n_int,
        "qi": [np.nan]*n_int,
        "qk": [np.nan]*n_int,
        "fb": [np.nan]*n_int,
        "tcf": [np.nan]*n_int,
        "tce": [np.nan]*n_int,
        "tce_velocity": [np.nan]*n_int,
        "tci": [np.nan]*n_int,
        "ttu": [np.nan]*n_int,
        "ttu_std": [np.nan]*n_int,
        "ttc": [np.nan]*n_int,
        "ttc_std": [np.nan]*n_int
    })

    print("\nCORE {0:03d} CALCULATING STATISTICS FOR EACH {1} INTERVAL".format(rank, params.int_size))

    for i in np.arange(n_int).tolist():

        int_start = starttime + i*pd.to_timedelta(params.int_size)
        int_end = (endtime + i*pd.to_timedelta(params.int_size))

        df.at[i, "Timestamp"] = int_start

        int_hr = mfi_df_hr[int_start:int_end]
        int_lr = int_hr.resample(params.dt_lr).mean()
        int_protons = df_protons[int_start:int_end]
        int_protons_lr = int_protons.resample(params.dt_lr).mean()

        # Record amount of missing data in each dataset in each interval
        if int_hr.empty:
            missing_mfi = 1
        else:
            missing_mfi = int_hr["Bx"].isna().sum()/len(int_hr)

        if int_protons.empty:
            missing_3dp = 1
        else:
            missing_3dp = int_protons["Vx"].isna().sum()/len(int_protons)

        # Save these values to their respective lists, to become columns in the final dataframe
        df.at[i, "missing_mfi"] = missing_mfi
        df.at[i, "missing_3dp"] = missing_3dp

        # What follows are nested if-else statements dealing with each combination of missing data between
        # the magnetic field and proton datasets, interpolating and calculating variables where possible
        # and setting them to missing where not 

        try: # try statement for error handling; see except statement at end of loop
            if missing_3dp <= 0.1:
                # Interpolate missing data, then fill any remaining gaps at start or end with nearest value
                int_protons = int_protons.interpolate(method="linear").ffill().bfill()
                int_protons_lr = int_protons_lr.interpolate(method="linear").ffill().bfill()

                # If any of the values are nan, nan will be returned OK
                df.at[i, "np"] = int_protons["np"].mean()
                df.at[i, "nalpha"] = int_protons["nalpha"].mean()
                df.at[i, "Tp"] = int_protons["Tp"].mean()
                df.at[i, "Talpha"] = int_protons["Talpha"].mean()

                Vx = int_protons["Vx"]
                Vy = int_protons["Vy"]
                Vz = int_protons["Vz"]

                Vx_mean = Vx.mean()
                Vy_mean = Vy.mean()
                Vz_mean = Vz.mean()

                # Save mean radial velocity (should dominate velocity mag)
                df.at[i, "v_r"] = np.abs(Vx_mean) # abs() because all vals negative (away from Sun)

                # Calculate velocity magnitude V0
                V0 = np.sqrt(Vx_mean**2+Vy_mean**2+Vz_mean**2)
                df.at[i, "V0"] = V0

                # Calculate rms velocity fluctuations, dv
                dvx = Vx - Vx_mean
                dvy = Vy - Vy_mean
                dvz = Vz - Vz_mean
                dv = np.sqrt(np.mean(dvx**2+dvy**2+dvz**2))
                df.at[i, "dv"] = dv

                # Compute proton velocity correlation length
                velocity_time_lags_lr, velocity_acf_lr = utils.compute_nd_acf(
                    np.array([
                        int_protons_lr.Vx,
                        int_protons_lr.Vy,
                        int_protons_lr.Vz
                    ]),
                    nlags=params.nlags_lr,
                    dt=float(params.dt_lr[:-1]))  # Removing "S" from end of dt string

                # velocity_acf_lr_list.append(velocity_acf_lr) #LOCAL ONLY
                velocity_corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(velocity_time_lags_lr, velocity_acf_lr)
                df.at[i, "tce_velocity"] = velocity_corr_scale_exp_trick

            if missing_mfi <= 0.1:

            # Interpolate missing data, then fill any remaining gaps at start or end with nearest value

                int_hr = int_hr.interpolate(method="linear").ffill().bfill() 
                int_lr = int_lr.interpolate(method="linear").ffill().bfill()

                # Firstly, do the calculations that we don't need proton data for
                # Then, condition on sufficient proton data and do the remainder

                # Resampling mag field data to 3s to match velocity data cadence
                ## NB: there is also a 3s cadence version of the mfi data, e.g. as used by Podesta2010, 
                ## but we want the highest res possible for the Taylor scale calculations

                Bx = int_hr["Bx"].resample(params.dt_protons).mean()
                By = int_hr["By"].resample(params.dt_protons).mean()
                Bz = int_hr["Bz"].resample(params.dt_protons).mean()

                Bx_mean = Bx.mean()
                By_mean = By.mean()
                Bz_mean = Bz.mean()

                # Calculate magnetic field magnitude B0
                B0 = np.sqrt(Bx_mean**2+By_mean**2+Bz_mean**2)
                df.at[i, "B0"] = B0

                # Calculate rms magnetic field fluctuations, db
                dbx = Bx - Bx_mean
                dby = By - By_mean
                dbz = Bz - Bz_mean
                db = np.sqrt(np.mean(dbx**2+dby**2+dbz**2))
                df.at[i, "db"] = db

                df.at[i, "dboB0"] = db/B0

                # Compute autocorrelations and power spectra
                time_lags_lr, acf_lr = utils.compute_nd_acf(
                    np.array([
                        int_lr.Bx,
                        int_lr.By,
                        int_lr.Bz
                    ]),
                    nlags=params.nlags_lr,
                    dt=float(params.dt_lr[:-1]))  # Removing "S" from end of dt string
                
                # acf_lr_list.append(acf_lr) #LOCAL ONLY

                corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(time_lags_lr, acf_lr)
                df.at[i, "tce"] = corr_scale_exp_trick

                # Use estimate from 1/e method to select fit amount
                corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(
                    time_lags_lr, acf_lr, np.round(2*corr_scale_exp_trick))
                df.at[i, "tcf"] = corr_scale_exp_fit

                corr_scale_int = utils.compute_outer_scale_integral(time_lags_lr, acf_lr)
                df.at[i, "tci"] = corr_scale_int

                time_lags_hr, acf_hr = utils.compute_nd_acf(
                    np.array([
                        int_hr.Bx,
                        int_hr.By,
                        int_hr.Bz
                    ]),
                    nlags=params.nlags_hr,
                    dt=float(params.dt_hr[:-1]))

                # acf_hr_list.append(acf_hr) #LOCAL ONLY

                # ~1min per interval due to spectrum smoothing algorithm
                # slope_i, slope_k, break_s = utils.compute_spectral_stats(
                #     np.array([
                #         int_hr.Bx,
                #         int_hr.By,
                #         int_hr.Bz
                #     ]),
                #     dt=float(params.dt_hr[:-1]),
                #     f_min_inertial=params.f_min_inertial, f_max_inertial=params.f_max_inertial,
                #     f_min_kinetic=params.f_min_kinetic, f_max_kinetic=params.f_max_kinetic)

                # df.at[i, "qi"] = slope_i
                # df.at[i, "qk"] = slope_k
                # df.at[i, "fb"] = break_s

                taylor_scale_u, taylor_scale_u_std = utils.compute_taylor_chuychai(
                    time_lags_hr,
                    acf_hr,
                    tau_min=params.tau_min,
                    tau_max=params.tau_max)

                df.at[i, "ttu"] = taylor_scale_u
                df.at[i, "ttu_std"] = taylor_scale_u_std

                # taylor_scale_c, taylor_scale_c_std = utils.compute_taylor_chuychai(
                #     time_lags_hr,
                #     acf_hr,
                #     tau_min=params.tau_min,
                #     tau_max=params.tau_max,
                #     q=slope_k)

                # df.at[i, "ttc"] = taylor_scale_c
                # df.at[i, "ttc_std"] = taylor_scale_c_std

            if missing_3dp <= 0.1 and missing_mfi <= 0.1:
                # Already interpolated data, shouldn't need to do again

                ## Convert magnetic field fluctuations to Alfvenic units
                alfven_prefactor = 21.8/np.sqrt(int_protons["np"]) # Converting nT to Gauss and cm/s to km/s
                # note that Wang2012ApJ uses the mean density of the interval

                dbx_a = dbx*alfven_prefactor
                dby_a = dby*alfven_prefactor
                dbz_a = dbz*alfven_prefactor
                db_a = np.sqrt(dbx_a**2+dby_a**2+dbz_a**2)
                db_a_rms = np.sqrt(np.mean(dbx_a**2+dby_a**2+dbz_a**2))
                df.at[i, "db_a"] = db_a_rms

                # Cross-helicity
                Hc = np.mean(dvx*dbx_a + dvy*dby_a + dvz*dbz_a)

                # Normalize by energy (should then range between -1 and 1, like a normal correlation coefficient)
                # Only minor different calculating them separately, rather than np.mean(dv**2 + db**2)
                e_kinetic = np.mean(dv**2)
                e_magnetic = np.mean(db_a**2)

                sigma_c = 2*Hc/(e_kinetic+e_magnetic)
                df.at[i, "sigma_c"] = sigma_c
                df.at[i, "sigma_c_abs"] = np.abs(sigma_c)
                # Normalized residual energy
                sigma_r = (e_kinetic-e_magnetic)/(e_kinetic+e_magnetic)
                df.at[i, "sigma_r"] = sigma_r

                # Alfven ratio (ratio between kinetic and magnetic energy, typically ~0.5)
                ra = e_kinetic/e_magnetic
                df.at[i, "ra"] = ra

                # Alignment cosine (see Parashar2018PRL, ranges between -1 and 1)
                cos_a = Hc/np.mean(np.sqrt(e_kinetic*e_magnetic))
                df.at[i, "cos_a"] = cos_a

                # Elsasser variables
                zpx = dvx + dbx_a
                zpy = dvy + dby_a
                zpz = dvz + dbz_a
                zp = np.sqrt(np.mean(zpx**2+zpy**2+zpz**2))
                df.at[i, "zp"] = zp

                zmx = dvx - dbx_a
                zmy = dvy - dby_a
                zmz = dvz - dbz_a
                zm = np.sqrt(np.mean(zmx**2+zmy**2+zmz**2))
                df.at[i, "zm"] = zm

        except Exception as e:
            print("Error: missingness < 10% but error in computations: {}".format(e))

    df = df.set_index("Timestamp")
    df = df.sort_index()

    # Merge electron data
    df = df.merge(df_electrons, how="left", left_index=True, right_index=True)

    # Calculating analytical, 12hr variables
    df["rhoe"] = 2.38*np.sqrt(df['Te'])/df['B0'] # Electron gyroradius
    df['rhop'] = 102*np.sqrt(df['Tp'])/df['B0'] # Ion gyroradius
    df["de"] = 5.31/np.sqrt(df["ne"]) # Electron inertial length
    df["dp"] = 228/np.sqrt(df["ne"]) # Ion inertial length
    df["betae"] = 0.403*df["ne"]*df["Te"]/(df["B0"]**2) # Electron plasma beta
    df["betap"] = 0.403*df["ne"]*df["Tp"]/(df["B0"]**2) # Ion plasma beta
    df["vte"] = 419*np.sqrt(df["Te"]) # Electron thermal velocity
    df["vtp"] = 9.79*np.sqrt(df["Tp"]) # Ion thermal velocity
    df["ms"] = df["V0"]/df["vtp"] # Sonic mach number (total speed)
    df["mst"] = df["dv"]/df["vtp"] # Sonic mach number (fluctuations)
    df["va"] = 21.8*df['B0']/np.sqrt(df["ne"]) # Alfven speed
    df["ma"] = df["V0"]/df["va"] # Alfven mach number (total speed)
    df["mat"] = df["dv"]/df["va"] # Alfven mach number (fluctuations)
    df["ld"] = 0.00743*np.sqrt(df["Te"])/np.sqrt(df["ne"]) # Debye length
    df["p"] = (2e-6)*df["ne"]*df["V0"]**2 # Proton ram pressure in nPa, from https://omniweb.gsfc.nasa.gov/ftpbrowser/bow_derivation.html
    df["fce"] = 28*df["B0"] # Electron gyrofrequency
    df["fci"] = 0.0152*df["B0"] # Ion gyrofrequency

    # Calculating Reynolds numbers (using pre-factors derived in paper)
    df["Re_lt"] = 27*(df["tcf"]/df["ttc"])**2
    df["Re_di"] = 2*((df["tcf"]*df["V0"])/df["dp"])**(4/3)
    df["tb"] = 1/((2*np.pi)*df["fb"])
    df["Re_tb"] = 2*((df["tcf"]/df["tb"]))**(4/3)

    # Converting scales from time to distance
    # (invoking Taylor's hypothesis)

    df['lambda_t_raw'] = df["ttu"]*df["V0"]
    df['lambda_t'] = df["ttc"]*df["V0"]
    df['lambda_c_e'] = df["tce"]*df["V0"]
    df['lambda_c_fit'] = df["tcf"]*df["V0"]
    df['lambda_c_int'] = df["tci"]*df["V0"]

    # Elsasser var decay rates
    df["zp_decay"] = (df["zp"]**3)/(df["lambda_c_fit"]) # Energy decay/cascade rate
    df["zm_decay"] = (df["zm"]**3)/(df["lambda_c_fit"]) # Energy decay/cascade rate

    #################################################################################

    print("\nFINAL DATAFRAME:\n")
    print(df.head())
    df.to_pickle("data/processed/" + date + ".pkl")

    # LOCAL: Plotting all ACFs to check calculations
    # for acf in acf_lr_list:
    #     plt.plot(acf)
    # plt.savefig("data/processed/all_acfs_lr.png")
    # plt.close()

    # for acf in velocity_acf_lr_list:
    #     plt.plot(acf)
    # plt.savefig("data/processed/all_proton_acfs_lr.png")
    # plt.close()

    # for acf in acf_hr_list:
    #     plt.plot(acf)
    # plt.savefig("data/processed/all_acfs_hr.png")
    # plt.close()

# wait until all parallel processes are finished
comm.Barrier()

if rank == 0:
    print("\n##################################")
    print("ALL CORES FINISHED")
