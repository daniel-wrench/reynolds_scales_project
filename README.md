# README
Codes for constructing a database of solar wind parameters and scales as measured by the *Wind* spacecraft and OMNI. It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the src/params.py parameter file.

Takes in 300GB of CDF files, produces 8MB CSV file. 

## To-do
Paper should be a story of how to calculate Re for the solar wind, including all the assumptions and annoyances along the way. 

2. Plots
    - Zenodo and new variables

    - *Fig. 4: Add multiple parabolic fit lines, put units at end of top 2 axes, add little arrows for vertical lines (fit, intercept)?*

1. Re-run codes on Rāpoi, checking my instructions

3. Refer to Cheng and wang paper
6. Perform checks in demo notebook with data from 1996, 2009, and 2021, compare with database

### Future work
- Thorough outlier and error analysis for both scales and the final Re estimate. Investigate anomalous slopes $q_k$. Check Excel and sort by these values to get problem timestamps. 
- Add decay rate $U^3/L$ (see eqn. 10 of Zhou2020, eqn. 1 of Wu2022), cross-helicity $\sigma_c$, residual energy $\sigma_r$, collisional age? (Kasper PRL)
- Add vector velocities to params.py script, anticipating switch to PSP data
- Think about using the standard error to express variation in the means of our Re estimates.
- More 2D histograms: 
    - Taylor scale vs. delta b/b
    - Tb vs. ion inertial timescale vs. Taylor scale. Cf. expectations reported in **Results** below

### References: 

- Three-part ApJ/ApJSS article on data product for studying *Electron Energy Partition across Interplanetary Shocks*. 
- Fordin2023: represents a cool use of using a very large Wind dataset for machine learning (classification) purposes.

## Notes on results

Poster and AGU talk are available in the Comms folder, paper draft is in process

### Outliers and data cleaning

- In `calculate_numerical_vars.py`, intervals with more than 10% missing data are removed completely (all values set to NA). Otherwise, any gaps are linearly interpolated. The average amount missing is 3%.


**2004-22 data**
- For 0.7% of timestamps, the slope of the inertial range is steeper than that of the kinetic range, which leads to strange, very small corrected values of the Taylor scale, and in turn very large values of Re_lt. There is also a value of very small uncorrected Taylor scales (waves? - potential for tangential study). I have made plots of these situations.

- Spectral breakscale frequencies seem to small, therefore timescales too big, therefore Re_tb too small. But indeed breakscale is still "a few times larger" than di, which is what we would expect (Leamon1998b)

### Correlations

#### Expected correlations:

- **Spectral breakscale:** From Bandy2020: *For example, Leamon et al. (2000, Fig. 4) and Wang et al. (2018) argued that the ion-inertial scale controls the spectral break and onset of strong dissipation, while Bruno & Trenchi (2014) suggested the break frequency is associated with the resonance condition for a parallel propagating Alfvén wave. Another possibility is that the largest of the proton kinetic scales terminates the inertial range and controls the spectral break (Chen et al. 2014).* See also Matthaeus2008, Fig. 3

- **Correlation scale vs. di:** see Cuesta2022 Fig. 2 and 3, note different variances of pdfs

- **$q_k$**: compare with delta b/b: Larger fluctuations causes larger decay rate, steeper slope q_k?, and temperature: Also, Leamon1998 describe correlation between temperature and the slopes of both the inertial and dissipation ranges. In general the temperature is of particular interest in correlating with other variables.

- **Solar cycle**: See Wicks2009 ApJ 690, Cheng2022 ApJ, Zhou2020Apj


## How to run this code

In order to create the full, 25-year dataset, an HPC cluster will be required. However, for the purposes of testing, a version of the pipeline is available that can be run locally on your machine with minimal computing requirements: note some Local/HPC differences in the instructions below.

**Google Colab** is a highly recommended way to run the code for beginners on a Windows computer. 
You will need to prefix the commands below with `!`, use `%cd` to move into the project folder, and can safely ignore step 2.

1. **Clone the repository to your local machine:**

    - Using a terminal: `git clone https://github.com/daniel-wrench/reynolds_scales_project`

    - [Using VS Code](https://learn.microsoft.com/en-us/azure/developer/javascript/how-to/with-visual-studio-code/clone-github-repository?tabs=create-repo-command-palette%2Cinitialize-repo-activity-bar%2Ccreate-branch-command-palette%2Ccommit-changes-command-palette%2Cpush-command-palette#clone-repository)

2. **Create and activate a virtual environment**: 

    (For HPCs, start with `module load python/3.9.5`. You may also need to use `python3` instead of `python`.)

    `python -m venv venv`

    `venv\Scripts\activate`

2. **Install the required packages:**

    `pip install -r requirements.txt`

3. **Download the raw CDF files using a set of recursive `wget` commands:**

    Local: `bash 0_download_files.sh`

    HPC: 
    - (`tmux new`)
    - `srun --pty --cpus-per-task=2 --mem=1G --time=01:00:00 --partition=quicktest bash`
    - `bash 0_download_files.sh`
    - (`Ctrl-b d` to detach from session, `tmux attach` to re-attach)

    There are approximately 10,000 files for each of the daily datasets (MFI, 3DP:PLSP and 3DP:ELM2), and 330 for the monthly datasets (OMNI). **Requries 300GB of disk space**.

4. **Get the raw variables by processing the CDF files:**

    Local: `bash 1_get_raw_vars_local.sh`

    HPC: `sbatch 1_get_raw_vars.sh`
        
        Recommended HPC job requirements: 
        This job can run on all the input files (1995-2022) with 256 CPUs/300GB/285min, but the following step cannot and uses all the data from this step, so recommended to instead run twice on half the data (:5000 and 5000:), with the line of code provided in the .py file, and use the following specifications: 256 CPUs/230GB/3 hours. (Takes about 8min/file/core).

    Process the raw CDF files, getting the desired variables at the desired cadences as specified in `params.py`. If more than 40% of values in any column are missing, skips that data file. Note that it is processing the mfi data that takes up the vast majority of the time for this step.

    NB: Missing data is not reported if you do not resample to the desired frequency first. It is important that we do note the amount of missing data in each interval, even if we do interpolate over it.

    For the non-mfi datasets, we only get a missing value for a 12-hour average if there is no data for that period. Otherwise, we simply get an average of the available data for each period. 

5. **Get the numerical variables by running a sequence of calculations:**

    Local: `bash 2_calculate_numerical_vars_local.sh` 

    HPC: `sbatch 2_calculate_numerical_vars.sh`
       
        Recommended HPC job requirements: 
        As stated in the previous step, this requires > 500GB of memory for the full dataset, so recommended to instead run twice on half the data (no change needed to the code, simply run previous step on half the data): 256 CPUS/320GB/5 hours.
        

    See the notebook **demo_scale_funcs.ipynb** for more on these calculations. Fitting parameters are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to.

6. **Get the analytical variables by running a sequence of calculations and output the final dataframe:**

    Local: `bash 3_calculate_analytical_vars_local.sh` 

    HPC: 
    - `srun --pty --cpus-per-task=1 --mem=1G --time=00:05:00 --partition=quicktest bash`
    - `bash 3_calculate_analytical_vars.sh`

The figures for the paper are produced in `4_plot_figures.py` and `demo_numerical_w_figs.ipynb`.

You will now find two output files corresponding to the final database and its summary statistics:

- `data/processed/wind_database.csv`
- `data/processed/wind_summary_stats.csv`

## Background

Previously, **Kevin de Lange** created a smaller version of this dataset and investigated the correlation between the Taylor scale and the other variables, including training machine learning models to predict the Taylor scale. *He found an unexpected, reasonably strong positive correlation between the Taylor scale and correlation scale*. Specifically, he found a **correlation of 0.77 between the Taylor scale and exponential-fit correlation scale**, and **0.62 between the Taylor scale and the 1/e-trick correlation scale** (see Figures 5.17 and 5.18 on page 57 of his thesis, *Mining Solar Wind Data for Turbulence Microscales*).                                                                 

We are now more rigorously estimating the Taylor scale to confirm or deny this correlation, which would have significant implications of a constant Reynolds number in the solar wind. At the same time, we are creating a database of many more years of data that also includes other parameters of interest such as plasma beta, gyroradii, etc.

## Data
Time series of physical parameters of the solar wind.

### Raw variables
From Wind spacecraft, where original cadence ~ 0.092S :
- Magnetic field strength (MFI H2)
- Proton density and temperature: (3DP PLSP)  
- Electron density and temperature (3DP ELM2)

From OMNI CDAWEB dataset, where original cadence = 1min:
- Flowspeed
- Pressure
- Magnetic field

See `doc/wind_database_metadata.xlsx` for more details, including description of the secondary variables such as ion inertial length and turbulence scales.

## Analysis
$Re=\frac{UL}{\nu}\approx(\frac{L}{\eta})^{4/3}\approx(\frac{L}{d_i})^{4/3}\approx(\frac{L}{\lambda_t})^2$

- $U$ is the flow speed
- $L$ is the characteristic length a.k.a. correlation scale: the size of the largest "eddies" at the start of inertial range of turbulence where energy is input into the system.
- $\nu$ is the kinematic viscosity, **which cannot be determined for a collisionless plasma**.
- $\eta$ is the Kolmogorov length scale at which eddies become critically damped and the "dissipation" range begins: $(\frac{\nu^3}{\epsilon})^{1/4}$, where $\epsilon$ is the rate of energy dissipation. We cannot determine this scale since we do not have viscosity, but we can approximate it using the ion inertial length $d_i$ and the spectral break $tb$.
- $\lambda_t$ is the Taylor microscale. This is the size of the smallest eddies in the inertial range, or the size of the largest eddies that are effected by dissipation. It is also related to mean square spatial derivatives, and can be intepreted as the "single-wavenumber equivalent dissipation scale" (Hinze 1975).

The Reynolds number represents the competition between inertial forces that promote turbulence against viscous forces that prevent it. It also represents the separation between the large, energy-containing length scale $L$ and the small, dissipative length scale $\eta$.

Therefore, using this dataset, we calculate the Reynolds number using the last two approximation in the equation above.
These calculations, along with statistics, correlations, and plots, are done in `4_analyse_results.py`.

- Summary stats, including correlations, of all variables, but mainly scales and Re
- Multi-var histograms and time series of scales and Re

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
