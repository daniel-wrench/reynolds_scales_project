#!/bin/bash -e

#SBATCH --job-name          1_process_data
#SBATCH --partition         parallel
#SBATCH --mem               150G
#SBATCH --cpus-per-task     256
#SBATCH --time              07:00:00
#SBATCH --output            %x_%j_12H.out
#SBATCH --error             %x_%j_12H.err
#SBATCH --mail-type         ALL
#SBATCH --mail-user         daniel.wrench@vuw.ac.nz

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load Python/3.10.4

source venv/bin/activate

echo "JOB STARTED"
date
echo -e "NB: Input files will appear out of order due to parallel processing. Output files will be in chronological order.\n"

# python src/process_sunspot_data.py
mpirun --oversubscribe -n 256 python src/process_data.py

echo "FINISHED"
