#!/bin/bash -e

#SBATCH --job-name          2_calculate_numerical_vars
#SBATCH --partition         parallel
##SBATCH --reservation	    spacejam
##SBATCH --nodelist          spj01
#SBATCH --mem		    320G
#SBATCH --cpus-per-task     256	
#SBATCH --time              5:00:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running in Raapoi
source ActivatePython.sh

echo "JOB STARTED"
date

mpirun --oversubscribe -n 256 python src/calculate_numerical_vars.py

date
