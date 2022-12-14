#!/bin/bash -e

#SBATCH --job-name          2_construct_database
#SBATCH --partition         parallel
##SBATCH --reservation       spacejam
##SBATCH --nodelist          spj01
#SBATCH --mem		    200G
#SBATCH --cpus-per-task     256	
#SBATCH --time              2:00:00
#SBATCH --output            %x.out
#SBATCH --error             %x.err

## For running in Raapoi
source ActivatePython.sh

echo "JOB STARTED"
date

mpirun --oversubscribe -n 256 python construct_database.py

date
