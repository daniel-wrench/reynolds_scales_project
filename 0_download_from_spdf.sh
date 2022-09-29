#!/bin/bash -e

## NEED TO INCREASE TIME FOR WIND MFI DATA ##

#SBATCH --job-name          0_download_from_spdf
#SBATCH --partition         quicktest
##SBATCH --nodelist          spj01
#SBATCH --mem-per-cpu       1G
#SBATCH --cpus-per-task     1
#SBATCH --time              00:15:00
#SBATCH --output            %x_%j.out
#SBATCH --error             %x_%j.err
#SBATCH --mail-type	    BEGIN, END, FAIL
#SBATCH --mail-user	    daniel.wrench@vuw.ac.nz

######## FOR BULK DOWNLOADING FROM NASA'S SPACE PHYSICS DATA FACILITY (CDAWEB) ########

# Use the following to download a sequence of files from a single directory
# (Currently downloading 1 week worth of data)
# wget --directory-prefix=data/raw/omni/omni_cdaweb/hro2_1min/2016/ https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/2016/omni_hro2_1min_20160101_v01.cdf
#wget --directory-prefix=data/raw/wind/3dp/3dp_elm2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/2016/wi_elm2_3dp_201601{01..07}_v02.cdf
#wget --directory-prefix=data/raw/wind/3dp/3dp_plsp/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_plsp/2016/wi_plsp_3dp_201601{01..07}_v02.cdf
#wget --directory-prefix=data/raw/wind/mfi/mfi_h2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/wi_h2_mfi_2016{11..12}{01..31}_v05.cdf

# Use the following to download all CDF files and sub-directories from a directory, removing the first two directories from the saved filepath
# In Raapoi terminal: 10.7MB/s
# In parallel partition: 13.7MB/s

wget --directory-prefix=data/raw/ --recursive -np -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/2016/
wget --directory-prefix=data/raw/ --recursive -np -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/2016/
wget --directory-prefix=data/raw/ --recursive -np -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_plsp/2016/
wget --directory-prefix=data/raw/ --recursive -np -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/

echo "FINISHED"
