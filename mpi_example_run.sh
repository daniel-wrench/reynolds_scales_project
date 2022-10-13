#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=parallel
##SBATCH --reservation=spacejam
#SBATCH --time=24:00:00
#SBATCH --job-name="ngrid4"
#SBATCH -o batch.out
#SBATCH -e batch.err

source ~/bin/python_env
cd /nfs/scratch/parashtu/DFN/ngrid4
mpirun --oversubscribe -n 64 python3 $HOME/WorkSpace/PIC-Distfn/distfn.py
