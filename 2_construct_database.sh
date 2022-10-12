#!/bin/bash -e

#SBATCH --job-name          2_construct_database
#SBATCH --partition         quicktest
##SBATCH --nodelist          spj01
#SBATCH --mem		    2G
#SBATCH --cpus-per-task     2
#SBATCH --time              00:15:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
source ActivatePython.sh

echo "JOB STARTED"
date

python construct_database.py

date
