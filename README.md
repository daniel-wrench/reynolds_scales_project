# README
Codes for constructing a database of solar wind parameters and scales as measured by the *Wind* spacecraft (and potentially other spacecraft too)

## To-do
1. Deal with issue with some timestamps in 95-02 database being ahead of the actual data, e.g.:
- 1998-07-01 in database is actually data from 1998-06-30 (1 day ahead)
- 2001-10-19 in database is actually data from 2001-10-17 (2 days ahead)
) 

    *Have made some changes to construct_database which should deal with this. Now running on 4 years of data 1995-1998 to check fix*

2. Deal with outliers in 95-02 dataset
- Most of the strange `ttc` values come from strange values of `qk`, where it is greater than `qi`. This occurs for 4% of timestamps in 95-02 dataset: see examples in `plots/`)
- However, there are also some very small `ttu` values, and a strange `tci`
3. See **Analysis** section below

### Optional next steps

- Add [sunspot number](https://www.sidc.be/silso/datafiles), probably in `3_merge_dataframes` step
- Add energies, decay rate (see eqn. 10 of Zhou2020, eqn. 1 of Wu2022)

## Background

Previously, **Kevin de Lange** created a smaller version of this dataset and investigated the correlation between the Taylor scale and the other variables, including training machine learning models to predict the Taylor scale. *He found an unexpected, reasonably strong positive correlation between the Taylor scale and correlation scale*. Specifically, he found a **correlation of 0.77 between the Taylor scale and exponential-fit correlation scale**, and **0.62 between the Taylor scale and the 1/e-trick correlation scale**.                                                                 

We are now more rigorously estimating the Taylor scale to confirm or deny this correlation. At the same time, we are creating a database of many more years of data that also includes other parameters of interest such as plasma beta, gyroradii, etc.

## Data
Time series of physical parameters of the solar wind.
For more details see the variable lists in `params.py`
- Wind spacecraft:
-- Magnetic field strength: MFI H2
-- Proton density and temperature: 3DP  
Re-sampled to two different frequencies and split into 12-hour intervals.

See `wind_database_metadata.xlsx` for description of variables.

Built using Python 3.9.5

## Analysis
$Re=\frac{UL}{\nu}=(\frac{L}{\lambda_t})^2=(\frac{L}{\eta})^{4/3}$

- $U$ is the flow speed
- $L$ is the characteristic length, a.k.a. correlation scale, at which energy is input into the system
- $\nu$ is the kinematic viscosity, **which cannot be determined for a collisionless plasma**.
- $\lambda_t$ is the Taylor microscale
- $\eta$ is the Kolmogorov length scale: $(\frac{\nu^3}{\epsilon})^{1/4}$. Given we cannot determine this, we can approximate it using the ion inertial length $d_i$ and the spectral break $db$
- $\epsilon$ is the rate of energy dissipation

Therefore, using this dataset, we can calculate the Reynolds number using the following three ways:

$Re=(\frac{L}{\lambda_t})^2=(\frac{L}{d_i})^{4/3}=(\frac{L}{db})^{4/3}$

Calculate correlations and plot relationships in `4_analyse_results.py`

## Pipeline
*NB*: The x_local.sh files are designed so test the cluster scripts locally: these can be run in your local terminal with the command `bash`.

### Setting up environment and downloading raw data
1. `module load python/3.9.5`
2. `python3 -m venv venv`
2. (`pip install --upgrade pip`)
2. `pip install -r requirements.txt`
2. (`tmux new`)
2. `srun --pty --cpus-per-task=2 --mem=1G --time=01:00:00 --partition=quicktest bash`
    
    `bash 0_download_from_spdf.sh`: Download the raw CDF files using a set of recursive wget commands. ~ 13MB/s, 5s per mfi file
2. (`Ctrl-b d` to detach from session, `tmux attach` to re-attach)

### Running scripts
1. `sbatch 1_process_raw_data.sh` **(12min/month using 10 cores, 1.5min/file running locally. 5.24GB: 1 month. 5.96GB: 2 months. 7.11GB: 4 months. 12.1GB?: 12 months)**: 
2.5 hours was not enough time for one year of data using 10 cores on parallel partition. 64 cores on quicktest gave an error.

Process the raw CDF files, getting the desired variables at the desired cadences as specified in `params.py`. If more than 40% of values in any column are missing, skips that data file. Note that it is processing the mfi data that takes up the vast majority of the time for this step.

2. `sbatch 2_construct_database.sh` **(12min/month using 10 cores. 5.44GB: 1 month. 6.61-6.65GB: 2 months. 8.87GB: 4 months)**: Construct the database, involving calculation of the analytically-derived and numerically-derived variables (see the notebook **demo_scale_funcs.ipynb** for more on these). Fitting parameters are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to. ****

3. `srun --pty --cpus-per-task=1 --mem=1G --time=00:05:00 --partition=quicktest bash`
    
    May need to adjust mem as datasize increases
    
4.  `bash 3_merge_dataframes.sh > 3_merge_dataframes.out`: Merge the final files into the database

### Kevin's old pipeline

1. `mfi_lr.py`: Make low-res (0.2Hz) magnetic field dataframe pickle (Rāpoi job)
2. `compute_corr.py` (Rāpoi job): Compute correlation scale using two methods for 6-hour intervals of low-res dataframe
    - 3D ACF is calculated for 1500 lags = 50min at 0.2Hz
    - Produces two estimates using `compute_correlation_scale()`:
        1. 1/e trick: `estimate` and `Correlation_timescale_est`, calculated using `estimate_correlation_scale()` *This method tends to produce larger values.*
        2. Exponential fit: `lambda_c` and `Correlation_timescale`, calculated using `curve_fit()`. `num_seconds_for_lambda_c_fit = 1000` is specified inside the parent function. *This method tends to produce smaller values.*
        3. *TO-DO: Integral scale*

3. `mfi_hr.py`: Make high-res (10Hz) magnetic field dataframe pickle (Rāpoi job)
4. `compute_taylor.py` (Rāpoi job): Compute Taylor scale using parabolic fit at origin for 6-hour intervals of high-res dataframe 
    - 3D ACF is calculated for 20 lags = 2s at 10Hz
    - `num_seconds_for_lambda_t_fit = 2` is specified inside the function `compute_taylor_time_scale()`

5. `electrons_6hr.py`: Make low-res (6hr) electron density dataframe pickle (Rāpoi job)
5. Run EDA, produce plots, train ML pipelines (other Jupyter notebooks which Kevin ran in Google Colab)
