# README
This repository contains a dataset of solar wind parameters and turbulence scales as measured by NASA's *Wind* spacecraft, as well as the software pipeline used to produce it. An explanation of both can be found below, including instructions for running the code yourself.

## To-do

1. Clean repo, names, outlier cleaning
2. Run on 3 years of data, note times for full 256 cores
3. Run on 12h, 8h, 4h, for full dataset
3. Talk through pipeline with Tulasi 
    - combine with master_stats fn in time series repo?
    - spectral smoothing bottleneck?
    - omni vars?
    - save raw and analytical values at native cadence? (currently do this with just the raw vars)
5. Update instructions here (copy docstring), pipeline diagram?
4. Merge with master branch
6. Publish new version of repo

---

5. Check how well new values match up against existing ones and literature, talk about with Tulasi (time scales, slopes and electron stats should all be the same, rest will be slightly different)
- https://pubs.aip.org/aip/pop/article/13/5/056505/1032771/Eddy-viscosity-and-flow-properties-of-the-solar : Table III for OMNI medians
- https://iopscience.iop.org/article/10.3847/1538-4365/ab64e6: Fig. 4 for PSP cos(theta), cross-helicity, residual energy
- https://iopscience.iop.org/article/10.1088/0004-637X/741/2/75/meta for ACE cos(theta) and cross-helicity
- Alfven mach number $\approx$ order 10, plasma beta $\approx$ 1
6. Merge and perform checks in demo notebook with data from 1996, 2009, and 2021, compare with database
7. Clean, subset, and calculate new stats and plot new figures
8. Check no. of points reported; make clear subset contains ... points

## Research output
Paper published in The Astrophysical Journal: *Statistics of Turbulence in the Solar Wind. I. What is the Reynolds Number of the Solar Wind?* This work examines multiple ways of calculating an effective Reynolds number of the solar wind, using a portion of the data from this dataset. A PDF of the article and a poster presented at the 2023 SHINE conference are available in the `doc/` folder. **Insert link**

Future work is planned to pursue data mining, examining trends and correlations between a variety of quantities. Some expected relationships are given below under the heading *References*.

## Data dictionary
The repository contains two datasets: 
- `wind_dataset.csv`, containing the full 28 years of data (1995-01-01 to 2022-12-31), and 
- `wind_dataset_l1_cleaned.csv`, a subset of the above, containing 18 years of data (2004-06-01 to 2022-12-31), during which the Wind spacecraft was situated at L1 (and therefore was not situated in the solar wind permanently rather than travelling in and out of the magnetosphere). **This dataset has also been cleaned of outliers.**

The table below gives a description of the variables found in both datasets. 
- Averages are are to 3sf
- rms = root-mean-square
- Angle brackets within equations also refer to 12-hour averages. 
- Formulae are adapted from NRL formulary, converting G to nT and cm to km. 
- $n_e$ is used in place of $n_p$ in derivations due to data issues.

Intermediate variables used in calculations:

- $\delta b_i= B_i-\langle B_i \rangle$
- $\delta b_{i,A}= \delta b_i(21.8/\sqrt{n_p})$
- $\delta v_i= v_i-\langle v_i \rangle$
- $z^{\pm}_i = \delta v_i \pm \delta b_{i,A}$
- $e_{kinetic}=\frac{1}{2}\langle |\delta v|^2 \rangle$
- $e_{magnetic}=\frac{1}{2}\langle |\delta b_A|^2 \rangle$

Column name | Symbol | Name | Mean value | Unit | Source/derivation |
| ------ | ------ | ---- | ---------- | ---- | ------ |
| missing_mfi | - | Fraction of missing MFI data | 0.01 | - | Wind: MFI H2 |
| missing_3dp | - | Fraction of missing 3DP data | 0.11 | - | Wind: 3DP PM |
| sn | - | Sunspot number | 56.3 | - | WDC-SILSO |
| ma | $M_a$ | Alfvén Mach number | 7.36 | - | $V_0/v_a$ |
| mat | $M_{a,t}$ | Alfvén Mach number of fluctuations | 0.4  | - | $\|\delta v\|/v_a$ |
| ms | $M_s$ | Sonic Mach number | 15.31 | - | $V_0/v_{T_p}$ |
| mst | $M_{s,t}$ | Sonic Mach number of fluctuations | 0.84 | - | $\|\delta b\|/v_{T_p}$ |
| betae | $\beta_e$ | Electron plasma beta | 0.82 | - | $0.403n_eT_e/B_0^2$ |
| betap | $\beta_p$ | Proton plasma beta | 0.53 | - | $0.403n_eT_p/B_0^2$ |
| sigma_c | $\sigma_c$ | Cross helicity | 0.01 | - | $\frac{\langle \delta v_x\delta b_{x,A}+\delta v_y\delta b_{y,A}+\delta v_z\delta b_{z,A}\rangle}{e_{kinetic}+e_{magnetic}}$ |
| sigma_c_abs | $\|\sigma_c\|$ | Cross helicity (absolute value) | - | - | $\|\sigma_c\|$ |
| sigma_r | $\sigma_R$ | Residual energy | -0.44 | - | $\frac{e_{kinetic}-e_{magnetic}}{e_{kinetic}+e_{magnetic}}$ |
| ra | $R_A$ | Alfv\'en ratio | 0.46 | - | $\frac{e_{kinetic}}{e_{magnetic}}$ |
| cos_a | $\cos(A)$ | Alignment cosine | 0.01 | - | $\frac{\langle \delta v_x\delta b_{x,A}+\delta v_y\delta b_{y,A}+\delta v_z\delta b_{y,A}\rangle}{\langle \sqrt{\|\delta v\| \|\delta b_A\|}\rangle}$ | | 
| qi | $q_i$ | Inertial range slope | -1.68 | - | Numerical method  |
| qk | $q_k$ | Kinetic range slope | -2.63 | - | Numerical method  |
| re_lt | $Re_{\lambda_t}$ | Reynolds number | 3,410,000 | - | $27(tcf/ttc)^2$ |
| re_di | $Re_{d_i}$ | Reynolds number | 330,000 | - | $2(tcfV_0/{d_i})^{4/3}$ |
| re_tb | $Re_{t_b}$ | Reynolds number | 116,000 | - | $2(tcf/tb)^{4/3}$ |
| fb |$f_b$ | Spectral break frequency | 0.27 | Hz | Numerical method  |
| fce | $f_{ce}$ | Electron gyrofrequency | - | Hz |  $28 \times B_0$ |
| fci | $f_{ci}$ | Ion gyrofrequency | - | Hz |  $0.0152 \times B_0$ |
| p | $p$ | Proton ram ressure | - | nPa | $(2e-6)n_eV_0^2$ |
| b0 | $B_0$ | Magnetic field magnitude | 6.01 | nT | $\sqrt{\langle B_x\rangle^2+\langle B_y\rangle^2+\langle B_z\rangle^2}$ |
| db | $\delta b$ | Magnetic field fluctuations (rms) | 3.83 | nT | $\sqrt{\langle \delta b_{x}^2+\delta b_{y}^2+\delta b_{z}^2\rangle}$ |
| dbob0 | $\delta b/B_0$ | Magnetic field fluctuations (normalized) | 0.71 | nT | $\delta b/B_0$ |
| ne | $n_e$ | Electron density | 4.18 | cm $^{-3}$ | Wind: 3DP ELM2 |
| np | $n_p$ | Proton density | 5.47 | cm $^{-3}$ | Wind: 3DP PM |
| nalpha | $n_\alpha$ | Alpha density | 0.14 | cm $^{-3}$ | Wind: 3DP PM |
| te | $T_e$ | Electron temperature | 13.9 | eV | Wind: 3DP ELM2 |
| tp | $T_p$ | Proton temperature | 15.4 | eV | Wind: 3DP PM |
| talpha | $T_\alpha$ | Alpha temperature | 63.8 | eV | Wind: 3DP PM |
| tb |$t_b$ | Spectral break time scale | 11.2 | s | $1/(2\pi f_b)$ |
| tcf | $\tau_C^\text{fit}$ | Correlation time scale (fit method) | 2160 | s | Numerical method  |
| tce | $\tau_C^\text{exp}$ | Correlation time scale (1/e method) | 2260 | s | Numerical method  |
| tci | $\tau_C^\text{int}$ | Correlation time scale (integral method) | 2090 | s | Numerical method  |
| tce_velocity | $\tau_{C,v}^\text{exp}$ | Correlation time scale (1/e method) for velocity |  | s | Numerical method  |
| ttu | $\tau_{TS}^\text{ext}$ | Taylor time scale (uncorrected) | 11.4 | s | Numerical method  |
| ttu_std | $\tau_{TS}^\text{ext}$ | Error of Taylor time scale (uncorrected) | 0.12 | s | Numerical method  |
| ttc | $\tau_{TS}$ | Taylor time scale (corrected) | 7.44 | s | $\tau_{TS}^\text{ext}\times$ Chuychai correction factor |
| ttc | $\tau_{TS}$ | Error of Taylor time scale (corrected) | 0.07 | s | $\tau_{TS}^\text{ext}\times$ Chuychai correction factor |
| rhoe |$\rho_e$ | Electron gyroradius | 1.78 | km | $2.38\sqrt{T_e}/B_0$ |
| rhop |$\rho_p$ | Proton gyroradius | 63.9 | km | $102\sqrt{T_p}/B_0$ |
| de |$d_e$ | Electron inertial length | 3.12 | km | $5.31/\sqrt{n_e}$ |
| dp | $d_p$ | Proton inertial length | 134 | km | $228/\sqrt{n_e}$ |
| ld |$l_d$ | Debye length | 0.02 | km | $0.00743\sqrt{T_e}/\sqrt{n_e}$ |
| lambda_c_fit | $\lambda_C^\text{fit}$ | Correlation length scale (fit method) | 899,000 | km | $\tau_C^\text{fit}\times V_0$  |
| lambda_c_exp | $\lambda_C^\text{exp}$ | Correlation length scale (1/e method) | 942,000 | km | $\tau_C^\text{exp}\times V_0$  |
| lambda_c_int | $\lambda_C^\text{int}$ | Correlation length scale (integral method) | 880,000 | km | $\tau_C^\text{int}\times V_0$  |
| lambda_t_raw | $\lambda_T^\text{ext}$ | Taylor length scale (uncorrected) | 4,770 | km | $\tau_{TS}^\text{ext}\times V_0$  |
| lambda_t | $\lambda_T$ | Taylor length scale (corrected) | 3,220 | km | $\tau_{TS}\times V_0$ |
| v0 | $V_0$ | Velocity magnitude (rms) | 439 | km/s | $\sqrt{\langle V_x\rangle^2+\langle V_y\rangle^2+\langle V_z\rangle^2}$ |
| vr |$V_r$ | Radial velocity | 438 | km/s | $\|V_x\|$ |
| dv | $\delta v$ | Velocity fluctuations (rms) | 26.2 | km/s | $\sqrt{\langle \delta v_x^2+\delta v_y^2+\delta v_z^2\rangle}$ |
| va | $v_A$ | Alfvén speed | 65.5 | km/s | $21.8B_0/\sqrt{n_e}$ |
| vte | $v_{T_e}$ | Electron thermal velocity | 1490 | km/s | $419\sqrt{T_e}$ |
| vtp | $v_{T_p}$ | Proton thermal velocity | 30.5 | km/s | $9.79\sqrt{T_p}$ |
| db_a | $\delta b_A$ | Magnetic field fluctuations (Alfven units, rms) | 42.4 | km/s | $\sqrt{\langle \delta b_{x,A}^2+\delta b_{y,A}^2+\delta b_{z,A}^2\rangle}$ |
| zp | $z^{+}$ | Positive Elsasser variable (rms) | 48.9 | km/s | $\sqrt{\langle {z^{+}_x}^2+{z^{+}_y}^2+{z^{+}_z}^2\rangle}$ |
| zm | $z^{-}$ | Negative Elsasser variable (rms) | 48.9 | km/s | $\sqrt{\langle {z^{-}_x}^2+{z^{-}_y}^2+{z^{-}_z}^2\rangle}$ |
| zp_decay | | Positive Elsasser variable decay rate | - | m^2/s^3 | ${z^{+}}**3/\lambda_C^\text{fit}$ |
| zm_decay | | Negative Elsasser variable decay rate | - | m^2/s^3 | ${z^{-}}**3/\lambda_C^\text{fit}$ |

### Data sources
Wind data is downloaded from NASA/GSFC’s Space Physics Data Facility (SPDF). The links to the repositories and metadata, used to download many files automatically, are given in `src/params.py`.
- Wind: NASA spacecraft launched in 1994 to study plasma processes in the near-Earth solar wind
    - 3DP: 3D plasma instument
        - ELM2: Electron moments, with raw cadence of 3s
        - PM: Proton and alpha particle moments, with raw cadence of 3s
    - MFI: Magnetic field instrument
        - H2: Magnetic field vector measurements, with raw cadence of 0.092s $B_x,B_y,B_z$
clark- WDC-SILSO: World Data Center repository of sunspot number dataset.

### Missing data
If there is more than 10% missing data for any of the consitutent time series for a given interval, the value for that interval is set to missing. Gaps smaller than 10% are handled with linear interpolation. Overall, we find

- 15% of proton data missing (and therefore in any derived vars). *Because of this and some anomalously small values, we use $n_e$ in place of $n_p$, as noted in the derivations above*
- 4% of magnetic field data missing
- 8% of electron data missing
- Between 17,000-20,000 points available for each variable, depending on % missing

### Comments

- Typical difference of 1-3% between B and V magnitudes calculated from Wind vs. OMNI values
- Notable differences pre-L1 for B and p
- 6 weird small values of V0 (~70km/s) throughout
- Spectral breakscale frequencies seem to small, therefore timescales too big, therefore Re_tb too small. But indeed breakscale is still "a few times larger" than di, which is what we would expect (Leamon1998b)
- A few very small values of ttu - waves? Potential for tangential study.
- (See plots in `plots/supplementary/`)

**Limiting to L1 dataset (June 2006-end 2022)**
- 5% of intervals (745) have qk shallower (greater than) -1.7. This leads to strange, very small values of ttc **these are removed in the cleaned dataset**.
- A further 1.5% (223) have qk shallower than qi (see supplementary plot). 
- (The full dataset also has )
- 6 values of negative tci, **these are removed in the cleaned dataset**
-**11-12,000 points for each variable in final cleaned dataset**



## How to run this code

It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the src/params.py parameter file.

The HPC version of the code currently ingests 300GB across 10,000 CDF files (data from 1995-2022) and produces an 18MB CSV file.

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
        This job can run on all the input files (1995-2022) with 256 CPUs/300GB/285min, but the following step cannot and uses all the data from this step, so recommended to instead run twice on half the data (:5000 and 5000:), with the line of code provided in the .py file, and use the following specifications: 256 CPUs/210GB/3 hours. (Takes about 8min/file/core).

    Process the raw CDF files, getting the desired variables at the desired cadences as specified in `params.py`. Saves resultant datasets to `data/processed/*spacecraft*/`
    
    If more than 40% of values in any column are missing, skips that data file. Note that it is processing the mfi data that takes up the vast majority of the time for this step.

    NB: Missing data is not reported if you do not resample to the desired frequency first. It is important that we do note the amount of missing data in each interval, even if we do interpolate over it.

    For the non-mfi datasets, we only get a missing value for a 12-hour average if there is no data for that period. Otherwise, we simply get an average of the available data for each period. 

5. **Get the numerical variables by running a sequence of calculations:**

    Local: `bash 2_calculate_numerical_vars_local.sh` 

    HPC: `sbatch 2_calculate_numerical_vars.sh`
       
        Recommended HPC job requirements: 
        As stated in the previous step, this requires > 500GB of memory for the full dataset, so recommended to instead run twice on half the data (no change needed to the code, simply run previous step on half the data): 256 CPUS/320GB/6 hours.
        

    See the notebook **demo_scale_funcs.ipynb** for more on these calculations. Fitting parameters are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to.

6. **Get the analytical variables by running a sequence of calculations and output the final dataframe:**

    Local: `bash 3_calculate_analytical_vars_local.sh > 3_calculate_analytical_vars_local.out` 

    HPC: `bash 3_calculate_analytical_vars.sh > 3_calculate_analytical_vars.out` 

The figures for the paper are produced in `5_plot_figures.py` and `demo_numerical_w_figs.ipynb`.

You will now find two output files corresponding to the final database and its summary statistics:

- `data/processed/wind_dataset.csv`
- `data/processed/wind_summary_stats.csv`

## Tracking dataset updates
- No longer using OMNI: deriving all variables from Wind data (but keeping them in temporarily for testing)
- Using 3DP/PM (science-quality 3s proton moments) instead of 3DP/PLSP (24s moments) in order to calculate things like cross helicity

## Future statistical analysis
- Interrogate lambda_T vs. dboB0 some more
- Comment on lack of small (<2000km) uncorrected Taylor scale values, compared with SmithEA2006? (See Bill's comments) 


- Note that for mfi data, it is more recent (2022) data that has version numbers less than 5
- This is not the case of 3dp data, which has large numbers of v02 and v05 data. For this data, v04 stands out as having high % missing: perhaps including all those with 100% missing.

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

- **Spectral breakscale:** Should be around 0.4-0.5Hz, corresponding to distances of 600-1000km (Weygand 2007) (our frequency is a bit low). Roberts2022 says 10s. From Bandy2020: *For example, Leamon et al. (2000, Fig. 4) and Wang et al. (2018) argued that the ion-inertial scale controls the spectral break and onset of strong dissipation, while Bruno | Trenchi (2014) suggested the break frequency is associated with the resonance condition for a parallel propagating Alfvén wave. Another possibility is that the largest of the proton kinetic scales terminates the inertial range and controls the spectral break (Chen et al. 2014).* See also Matthaeus2008, Fig. 3; Vech2018. Leamon et al. 

- **Correlation scale vs. di:** see Cuesta2022 Fig. 2 and 3, note different variances of pdfs

- **$q_k$**: Expect to be about -8/3 (-2.67) Compare with delta b/b: Larger fluctuations causes larger decay rate, steeper slope q_k?, and temperature: Also, Leamon1998 describe correlation between temperature and the slopes of both the inertial and dissipation ranges. In general the temperature is of particular interest in correlating with other variables.

- **Solar cycle**: See Wicks2009 ApJ 690, Cheng2022 ApJ, Zhou2020Apj

## Background

Previously, **Kevin de Lange** created a smaller version of this dataset and investigated the correlation between the Taylor scale and the other variables, including training machine learning models to predict the Taylor scale. *He found an unexpected, reasonably strong positive correlation between the Taylor scale and correlation scale*. Specifically, he found a **correlation of 0.77 between the Taylor scale and exponential-fit correlation scale**, and **0.62 between the Taylor scale and the 1/e-trick correlation scale** (see Figures 5.17 and 5.18 on page 57 of his thesis, *Mining Solar Wind Data for Turbulence Microscales*).                                                                 

We are now more rigorously estimating the Taylor scale to confirm or deny this correlation, which would have significant implications of a constant Reynolds number in the solar wind. At the same time, we are creating a database of many more years of data that also includes other parameters of interest such as plasma beta, gyroradii, etc.