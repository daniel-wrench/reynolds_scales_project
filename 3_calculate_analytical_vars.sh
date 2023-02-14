#!/bin/bash -e

# Run this in an interactive session

source ActivatePython.sh

echo "JOB STARTED"
date

python src/calculate_analytical_vars.py

date
echo "FINISHED"