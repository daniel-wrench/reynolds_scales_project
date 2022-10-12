#!/bin/bash -e

source ActivatePython.sh

echo "JOB STARTED"
date

python merge_dataframes.py omni_path        int_size
python merge_dataframes.py electron_path    int_size
python merge_dataframes.py proton_path      int_size
python merge_dataframes.py mag_path         dt_hr
python merge_dataframes.py mag_path         dt_lr

echo "FINISHED"