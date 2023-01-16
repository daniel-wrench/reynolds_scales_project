# README
Codes for constructing a database of solar wind parameters and scales as measured by the *Wind* spacecraft (and potentially other spacecraft too)

## To-do
1. Tidy repo
5. Perform checks in demo notebook with data from 1996, 2009, and 2021, compare with database
5. Calculate summary stats, put in metadata, try to figure out tb issue (compare tb with Pitna, try to sort out to get decent Re numbers)
2. Plots
- Expand upon introduction
- Make corr scale intro. plots a 3-panel plot, annotate with values
- Make taylor scale intro plots a 2-panel plot
- Add di to the power spectrum
- Put Re table at the end, discussion of scales before then
- Get streamlined results from Rāpoi codes
- Investigate spectral break scale. Add di, tb (calculate their ratio) and other scales to the power spectrum in the demo notebook, ala Fig. 5 in Phillips2022. We are assuming our frequency ranges are in line with the work of Pitna.
- Look into outlier and error analysis for both scales and the final Re estimate. Note the skewed distribution we have to deal with, and Bill’s point that the correlation scale is known to have a log-normal distribution.


### Optional next steps

- Add [sunspot number](https://www.sidc.be/silso/datafiles), probably in `3_merge_dataframes` step
- Add energies, decay rate (see eqn. 10 of Zhou2020, eqn. 1 of Wu2022)
- Once Re calculations have been finalised, put into `construct_database.py`

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

See `wind_database_metadata.xlsx` for more details, including description of the secondary variables such as ion inertial length and turbulence scales.

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

## Results

### Outliers
1995-98 data: for 0.7% of timestamps, the slope of the inertial range is steeper than that of the kinetic range, which leads to strange, very small corrected values of the Taylor scale, and in turn very large values of Re_lt. There is also a value of very small uncorrected Taylor scales (waves? - potential for tangential study). I have made plots of these situations.

### Correlations
tc vs. di: see Cuesta2022 Fig. 2 and 3, note different variances of pdfs

### AGU talk
- See Comms folder

## Pipeline
*NB*: The x_local.sh files are designed so test the cluster scripts locally: these can be run in your local terminal with the command `bash`.
Built using Python 3.9.5

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

NB: Missing data is not reported if you do not resample to the desired frequency first. It is important that we do note the amount of missing data in each interval, even if we do interpolate over it.

For the non-mfi datasets, we only get a missing value for a 12-hour average if there is no data for that period. Otherwise, we simply get an average of the available data for each period. 


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
