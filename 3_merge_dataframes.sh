#!/bin/bash -e

# Run this in an interactive session

source ActivatePython.sh

echo "JOB STARTED"
date

python src/merge_dataframes.py

date
echo "FINISHED"