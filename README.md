# README

## Codes for calculating and analysing the Taylor scale, correlation scale, and other solar wind parameters

Basic workflow is demonstrated in `Summary.ipynb` (best to run in Google Colab as quite slow on local machine)

Helper functions are stored in `utils.py`

# Data
Magnetic field from Wind spacecraft, 2016-2020, using CDAWeb FTP and `cdflib`: WI_H2_MFI
Re-sampled to two different frequencies and split into 6-hour intervals.


# Workflow

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

# Results for scales and correlations

- In terms of the two methods for calculating the correlation scale, 1/e tends to produce larger values than the fit method, and this discrepancy becomes more pronounced (scatterplot shows greater scatter) at higher values of each. Overall only **0.88** correlation between the two values.

- Correlation between Taylor scale and exponential fit correlation scale = **0.77**. A time series plot of these quantities over time also shows a strong correlation.
- Correlation between Taylor scale and 1/e correlation scale = **0.62**. A time series plot of these quantities over time also shows a correlation, though weaker than that above.

# V2, implementing Chuychai

- Calculate *cross*-correlation, in addition to Pearson correlation coefficient? Probably more appropriate for time series data
- Use more data?