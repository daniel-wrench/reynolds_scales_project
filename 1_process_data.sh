#!/bin/bash -e

#SBATCH --job-name          1_process_data
#SBATCH --partition         parallel
#SBATCH --mem               10G
#SBATCH --cpus-per-task     3
#SBATCH --time              00:10:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

source ActivatePython.sh

echo "JOB STARTED"
date
echo -e "NB: Input files will appear out of order due to parallel processing. Output files will be in chronological order.\n"

# python src/process_sunspot_data.py
mpirun --oversubscribe -n 3 python src/process_data_NEW.py

echo "FINISHED"
