#!/bin/bash -e

#SBATCH --job-name          1_process_raw_data
#SBATCH --partition         quicktest
##SBATCH --nodelist          spj01
#SBATCH --mem-per-cpu       4G
#SBATCH --cpus-per-task     2
#SBATCH --time              00:30:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
module load python/3.8.1
source venv/bin/activate

echo "JOB STARTED"
date

#python process_raw_data.py omni_path omni_vars omni_thresh int_size
#python process_raw_data.py electron_path electron_vars electron_thresh int_size
#python process_raw_data.py proton_path proton_vars proton_thresh int_size

# 4min and 100MB per day/file
python process_raw_data.py mag_path mag_vars mag_thresh dt_hr

# 4min and 2MB per day/file (but 50MB when reading in raw data)
python process_raw_data.py mag_path mag_vars mag_thresh dt_lr

echo "FINISHED"
