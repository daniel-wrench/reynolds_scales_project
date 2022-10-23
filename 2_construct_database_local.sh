#!/bin/bash -e

source venv/Scripts/activate

echo "JOB STARTED"
date

python construct_database_local.py

date
