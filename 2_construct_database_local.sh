#!/bin/bash -e

source venv/Scripts/activate

echo "JOB STARTED"
date

python src/construct_database_local.py

date
