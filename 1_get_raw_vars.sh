#!/bin/bash -e

#SBATCH --job-name          1_get_raw_vars
##SBATCH --partition         parallel
#SBATCH --reservation       spacejam
##SBATCH --nodelist  	     spj01
#SBATCH --mem               230G
#SBATCH --cpus-per-task     256
#SBATCH --time              03:00:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
source ActivatePython.sh

echo "JOB STARTED"
date
echo -e "NB: Input files will appear out of order due to parallel processing. Output files will be in chronological order.\n"

python src/process_sunspot_data.py
mpirun --oversubscribe -n 64  python src/get_raw_vars.py omni_path        omni_vars       omni_thresh     int_size    None
mpirun --oversubscribe -n 256 python src/get_raw_vars.py electron_path    electron_vars   electron_thresh int_size    None
mpirun --oversubscribe -n 256 python src/get_raw_vars.py proton_path      proton_vars     proton_thresh   int_size    None
mpirun --oversubscribe -n 256 python src/get_raw_vars.py mag_path         mag_vars        mag_thresh      dt_hr       dt_lr

echo "FINISHED"
