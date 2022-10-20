# README
Codes for constructing a database of solar wind parameters and scales as measured by the *Wind* spacecraft (and potentially other spacecraft too)

**The pipeline works for January 2016, and in decent time too. Congrats! Now to fix an issue with missing data in February before running on lots of data.**

## To-do

1. Test pipeline in Raapoi on 1 year of data (2016). (You can keep codes in home but data in scratch, using $HOME call: see `example_run.sh`) **Have downloaded all raw data.**
- Change my scripts to resample after combining and check for missing data then, and do not interpolate gaps YET 

*You need to resample after combining the dataframes. If you resample beforehand, you will not pick up on missing data until you resample to a lower frequency; values far apart in time will simply sit next to each other in the dataframe, with no NAs in between.*
- Change 2_ to also add the % data missing for each interval as an output in the final dataframe. Then, interpolate if < 20% data missing (without doing this, the pipeline will happily run up to calculation of correlation scales.) 
- (Test all this locally)
- Make construct_database.py parallel (simply run whole thing on each of the outputs of 1_process_raw_data.py: for given rank, read the corresponding file. May also need to put timestamps in the output file names of 1_... to be able to identify the files for a given time period to review min/max scale calculations)
2. Check output plots against summary stats
2. Run pipeline on as much data as possible.
2. Spearman correlation?
2. Add energies?
2. Once database and correlations are produced, reflect on next steps: do I work more with this data, or switch back to looking at the missing data problem?

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

## Pipeline
*NB*: The .sh files are designed so that they can be tested locally (on a much reduced set of data): simply switch to the appropriate venv command in the file and then change the command to run the file from `sbatch` to `bash`.

### Setting up environment and downloading raw data
1. `module load python/3.9.5`
2. `python3 -m venv venv`
2. (`pip install --upgrade pip`)
2. `pip install -r requirements.txt`
2. (`tmux new`)
2. `srun --pty --cpus-per-task=2 --mem=1G --time=01:00:00 --partition=quicktest bash`
    
    May want to run in another partition to get faster speeds
    
    `bash 0_download_from_spdf.sh`: Download the raw CDF files using a set of recursive wget commands.
2. (`Ctrl-b, d`)

### Running scripts
1. `sbatch 1_process_raw_data.sh`: Process the raw CDF files, getting the desired variables at the desired cadences as specified in `params.py`. **Takes ~???min/file running locally.** If more than 40% of values in any column are missing, skips that data file. **Run with memory ~ 6GB/month of data**

2. `sbatch 2_construct_database.sh`: Construct the database, involving calculation of the analytically-derived and numerically-derived variables (see the notebook **demo_scale_funcs.ipynb** for more on these). Fitting parameters are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to.

3. `srun --pty --cpus-per-task=1 --mem=1G --time=00:05:00 --partition=quicktest bash`
    
    May need to adjust mem as datasize increases
    
4.  `bash 3_merge_dataframes.sh > 3_merge_dataframes.out`: Merge the final files into the database
---

## Kevin's old pipeline

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
