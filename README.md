# README
Codes for constructing a database of solar wind parameters and scales as measured by the *Wind* spacecraft, with optional merging with OMNI data. It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the src/params.py parameter file.

Currently ingests 300GB of CDF files (data from 1995-2022) and produces a 8MB CSV file.

## Research output
- Paper submitted to ApJ, entitled *Statistics of Turbulence in the Solar Wind. I. What is the Reynolds Number of the Solar Wind?* Looks at multiple ways of calculating the Reynolds number for a large solar wind dataset. Poster and AGU talk are available in the Comms folder. **Insert link**

## Current dataset
- Missing some rhoe, rhoi, beta, va (should only correspond to B0 missingness)
- Strange values in March 1996
- **Does not use ne in place of ni; backup in data/processed does**
- **Need to update prefactors**

## To-do
1. **Complete response to referee (see word doc), send off to co-authors with updated manuscript by Monday**
4. Run step 1 on [:5000]
5. Add thermal velocities
5. Run steps 2-3 
5. Check ni issue (see `plots/spare`)
6. Check issue with too many missing rhoe, rhoi, beta, va SHOULD CORRELATE TO B0 MISSINGNESS, and also strange values in March 96
6. Check how well new values match up against existing ones and literature, talk about with Tulasi (time scales, slopes and electron stats should all be the same, rest will be slightly different)
- NB: Final dataset in this directory does not use ne in place of ni
- https://pubs.aip.org/aip/pop/article/13/5/056505/1032771/Eddy-viscosity-and-flow-properties-of-the-solar: Table III for OMNI medians
- https://iopscience.iop.org/article/10.3847/1538-4365/ab64e6: Fig. 4 for PSP cos(theta), cross-helicity, residual energy
- https://iopscience.iop.org/article/10.1088/0004-637X/741/2/75/meta for ACE cos(theta) and cross-helicity
7. Once confirmed, run on [:5000]
8. Merge and perform checks in demo notebook with data from 1996, 2009, and 2021, compare with database
11. Clean, subset, and calculate new stats and plot new figures
12. Use standard error instead of SD?
13. Check no. of points reported; make clear subset contains ... points

## Tracking dataset updates
- No longer using OMNI: deriving all variables from Wind data (but keeping them in temporarily for testing)
- Using 3DP/PM (science-quality 3s proton moments) instead of 3DP/PLSP (24s moments) in order to calculate things like cross helicity
- Added new variables such as cross-helicity and elsasser decay rates
- Calculating db/B0 slightly differently
- Previous dataset had mistake of Bwind being retained despite high missing %

-np avg =8.6, vs. 8.3
-tp = 11.9, vs. 15.8
- CALCULATE % DIFFERENCE B0 VS B0MNI, dboB0 vs. db/Bwind (from previous dataset), V0 vs. vomni vs. v_r,
pomni vs. p

## Outliers and data cleaning

- In `calculate_numerical_vars.py`, intervals with more than 10% missing data are removed completely (all values set to NA). Otherwise, any gaps are linearly interpolated. The average amount missing is 3%.

**2004-22 data**
- For 0.7% of timestamps, the slope of the inertial range is steeper than that of the kinetic range, which leads to strange, very small corrected values of the Taylor scale, and in turn very large values of Re_lt. There is also a value of very small uncorrected Taylor scales (waves? - potential for tangential study). I have made plots of these situations.

- Spectral breakscale frequencies seem to small, therefore timescales too big, therefore Re_tb too small. But indeed breakscale is still "a few times larger" than di, which is what we would expect (Leamon1998b)

## Future statistical analysis
- Thorough outlier and error analysis for both scales and the final Re estimate. Investigate anomalous slopes $q_k$. Check Excel and sort by these values to get problem timestamps. 

- Add vector velocities to params.py script, anticipating switch to PSP data
- Think about using the standard error to express variation in the means of our Re estimates.
- More 2D histograms:
    - Taylor scale, breakscale vs. delta b/b
    - qk vs. delta b/b -> removing shallow qk likely removes small delta b/b due to ion lions: *larger fluctuations = more energy goes to ions (lions) = less energy for electrons (hyenas) = less power at electron (subion) scales = steeper slope*
    - Tb vs. ion inertial timescale vs. Taylor scale. Cf. expectations reported in **Results** below

## References

- Three-part ApJ/ApJSS article on data product for studying *Electron Energy Partition across Interplanetary Shocks*. 
- Fordin2023: represents a cool use of using a very large Wind dataset for machine learning (classification) purposes.
- Podesta2010 used a large Wind dataset, mostly for calculating a range of different power spectra, including of cross-helicity

### Expected correlations

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

    `venv\Scripts\activate` (For HPCs, use `source venv/bin/activate`)

2. **Install the required packages:**

    `pip install -r requirements.txt`

3. **Download the raw CDF files using a set of recursive `wget` commands:**

    Local: `bash 0_download_files.sh`

    HPC: 
    - (`tmux new`)
    - `srun --pty --cpus-per-task=2 --mem=1G --time=02:00:00 --partition=quicktest bash`
    - `bash 0_download_files.sh`
    - (`Ctrl-b d` to detach from session, `tmux attach` to re-attach)

    There are approximately 10,000 files for each of the daily datasets (MFI, 3DP:PLSP and 3DP:ELM2), and 330 for the monthly datasets (OMNI). **Requries 300GB of disk space**.

4. **Get the raw variables by processing the CDF files:**

    Local: `bash 1_get_raw_vars_local.sh`

    HPC: `sbatch 1_get_raw_vars.sh`
        
        Recommended HPC job requirements: 
        This job can run on all the input files (1995-2022) with 256 CPUs/300GB/285min, but the following step cannot and uses all the data from this step, so recommended to instead run twice on half the data (:5000 and 5000:), with the line of code provided in the .py file, and use the following specifications: 256 CPUs/230GB/3 hours. (Takes about 8min/file/core).

    Process the raw CDF files, getting the desired variables at the desired cadences as specified in `params.py`. Saves resultant datasets to `data/processed/*spacecraft*/`
    
    If more than 40% of values in any column are missing, skips that data file. Note that it is processing the mfi data that takes up the vast majority of the time for this step.

    NB: Missing data is not reported if you do not resample to the desired frequency first. It is important that we do note the amount of missing data in each interval, even if we do interpolate over it.

    For the non-mfi datasets, we only get a missing value for a 12-hour average if there is no data for that period. Otherwise, we simply get an average of the available data for each period. 

5. **Get the numerical variables by running a sequence of calculations:**

    Local: `bash 2_calculate_numerical_vars_local.sh` 

    HPC: `sbatch 2_calculate_numerical_vars.sh`
       
        Recommended HPC job requirements: 
        As stated in the previous step, this requires > 500GB of memory for the full dataset, so recommended to instead run twice on half the data (no change needed to the code, simply run previous step on half the data): 256 CPUS/320GB/5 hours.
        

    See the notebook **demo_scale_funcs.ipynb** for more on these calculations. Fitting parameters are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to.

6. **Get the analytical variables by running a sequence of calculations and output the final dataframe:**

    Local and HPC: `bash 3_calculate_analytical_vars_local.sh > 3_calculate_analytical_vars_local.out` 

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