# README

## Codes for calculating and analysing the Taylor scale, correlation scale, and other solar wind parameters

Basic workflow is demonstrated in `Summary.ipynb` (best to run in Google Colab as quite slow on local machine)

Helper functions are stored in `utils.py`

## Data
Magnetic field from Wind spacecraft, 2016-2020, using CDAWeb FTP and `cdflib`: WI_H2_MFI
Re-sampled to two different frequencies and split into 12-hour intervals.

See `metadata.xlsx` for description of variables.

## New workflow

1. `sbatch 1_process_raw_data.sh`: Process the raw CDF files, getting the desired variables at the desired cadences.
    - This same bash script can also be ran locally (on a small amount of data)
    - If more than 40% of values in any column are missing, skip data period 

## Kevin's workflow

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
