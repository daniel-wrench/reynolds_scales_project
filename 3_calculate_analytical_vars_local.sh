#!/bin/bash -e

source venv/Scripts/activate

echo "JOB STARTED"
date

python src/calculate_analytical_vars_local.py

date
echo "FINISHED"