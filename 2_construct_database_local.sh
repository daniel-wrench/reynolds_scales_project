#!/bin/bash -e

source venv/Scripts/activate

echo "JOB STARTED"
date

python construct_database.py

date
