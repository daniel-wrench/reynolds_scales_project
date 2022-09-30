#!/bin/bash -e

#SBATCH --job-name          2_construct_database
#SBATCH --partition         parallel
##SBATCH --nodelist          spj01
#SBATCH --mem-per-cpu       2G
#SBATCH --cpus-per-task     2
#SBATCH --time              04:00:00
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
