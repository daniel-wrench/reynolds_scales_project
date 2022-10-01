#!/bin/bash -e

#SBATCH --job-name          1_process_raw_data
#SBATCH --partition         quicktest
##SBATCH --nodelist          spj01
#SBATCH --mem-per-cpu       1G
#SBATCH --cpus-per-task     1
#SBATCH --time              01:00:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
module load python/3.8.1
source venv/bin/activate

python process_raw_data.py omni_path omni_vars omni_thresh int_size
python process_raw_data.py electron_path electron_vars electron_thresh int_size
python process_raw_data.py proton_path proton_vars proton_thresh int_size

# 3min per day to run on 5 cores
# 200MB per day in final output file
python process_raw_data.py mag_path mag_vars mag_thresh dt_hr

# 3min per day to run on 5 cores, 300MB memory utilised
# 0.4MB per day in final output file
python process_raw_data.py mag_path mag_vars mag_thresh dt_lr

echo "FINISHED"
