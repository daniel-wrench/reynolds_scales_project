#!/bin/bash -e

#SBATCH --job-name          2_construct_database
#SBATCH --partition         quicktest
##SBATCH --nodelist          spj01
#SBATCH --mem-per-cpu       1G
#SBATCH --cpus-per-task     1
#SBATCH --time              00:15:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err
#SBATCH --mail-type	    BEGIN, END, FAIL
#SBATCH --mail-user	    daniel.wrench@vuw.ac.nz

## For running locally
#source venv/Scripts/activate

## For running in Raapoi
module load python/3.8.1
source venv/bin/activate

python construct_database.py

echo "FINISHED"
