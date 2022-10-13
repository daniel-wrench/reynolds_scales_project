#!/bin/bash -e

#SBATCH --job-name          2_construct_database
#SBATCH --partition         quicktest
##SBATCH --partition         parallel
##SBATCH --reservation       spacejam
##SBATCH --node              spj01
#SBATCH --mem		        4G
#SBATCH --cpus-per-task     8
#SBATCH --time              00:30:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running in Raapoi
source ActivatePython.sh

echo "JOB STARTED"
date

mpirun --oversubscribe -n 8 python construct_database.py

date
