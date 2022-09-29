#!/bin/bash -e

#SBATCH --job-name          1_process_raw_data
#SBATCH --partition         parallel
##SBATCH --nodelist          spj01
#SBATCH --mem-per-cpu       1G
#SBATCH --cpus-per-task     1
#SBATCH --time              01:00:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err
#SBATCH --mail-type	    BEGIN, END, FAIL
#SBATCH --mail-user	    daniel.wrench@vuw.ac.nz

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
module load python/3.8.1
source venv/bin/activate

python process_data_omni.py
python process_data_wind_electrons.py
python process_data_wind_protons.py
python process_data_wind_mfi_hr.py
python process_data_wind_mfi_lr.py

echo "FINISHED"
