#!/bin/bash -e

source venv/Scripts/activate

echo "JOB STARTED"
date

python process_raw_data_local.py omni_path        omni_vars       omni_thresh     int_size    None
python process_raw_data_local.py electron_path    electron_vars   electron_thresh int_size    None
python process_raw_data_local.py proton_path      proton_vars     proton_thresh   int_size    None
python process_raw_data_local.py mag_path         mag_vars        mag_thresh      dt_hr       dt_lr

echo "FINISHED"
