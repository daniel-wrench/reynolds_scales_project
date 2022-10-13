#!/bin/bash -e

#SBATCH --job-name          1_process_raw_data
#SBATCH --partition         quicktest
##SBATCH --nodelist         spj01
#SBATCH --mem               2G
#SBATCH --cpus-per-task     4
#SBATCH --time              00:15:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
source ActivatePython.sh

echo "JOB STARTED"
date
echo -e "NB: Input files will appear out of order due to parallel processing. Output files will be in chronological order.\n"

mpirun --oversubscribe -n 4 python process_raw_data.py omni_path        omni_vars       omni_thresh     int_size    None
mpirun --oversubscribe -n 4 python process_raw_data.py electron_path    electron_vars   electron_thresh int_size    None
mpirun --oversubscribe -n 4 python process_raw_data.py proton_path      proton_vars     proton_thresh   int_size    None
# Issue with the following line. 0.092_{rank}.pkl files contain 5S data for the first 4 days, 0.092S for the last day
# Attempting fix in process_data.py
mpirun --oversubscribe -n 4 python process_raw_data.py mag_path         mag_vars        mag_thresh      dt_hr       dt_lr

echo "FINISHED"
