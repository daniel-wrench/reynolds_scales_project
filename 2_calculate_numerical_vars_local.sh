#!/bin/bash -e

source venv/Scripts/activate

echo "JOB STARTED"
date

python src/calculate_numerical_vars_local.py

date
