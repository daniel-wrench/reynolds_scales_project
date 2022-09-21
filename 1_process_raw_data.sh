#!/bin/bash -e

#SBATCH --job-name          1_process_raw_data
#SBATCH --partition         parallel
#SBATCH --nodelist          c01n01
#SBATCH --reservation       spacejam
#SBATCH --mem-per-cpu       1G
#SBATCH --cpus-per-task     1
#SBATCH --time              00:15:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running locally
source venv/Scripts/activate

## For running in Raapoi
#source venv/bin/activate

#python process_data_wind_electrons.py
#python process_data_wind_protons.py
python process_data_omni.py
# python process_data_wind_mfi_hr.py
# python process_data_wind_mfi_lr.py

echo "FINISHED"
