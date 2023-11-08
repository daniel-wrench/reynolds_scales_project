#!/bin/bash -e

######## FOR BULK DOWNLOADING FROM NASA'S SPACE PHYSICS DATA FACILITY (CDAWEB) ########

# Currently run in an interactive Rāpoi session using tmux
# (can also run locally on a small number of files)

# Get solar cycle (sunspot) data from SIDC (Solar Influences Data Analysis Center)
# echo "Downloading sunspot data..."
# wget --no-clobber --directory-prefix=data/raw/sunspots https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt
# echo "Sunspot data downloaded."

# Download a sequence of files from a specific directory
## (Currently downloading 1 week worth of data: takes about 4min locally, 20s on Google Colab)

# echo "Downloading OMNI data"
# wget --no-clobber --directory-prefix=data/raw/omni/omni_cdaweb/hro2_1min/2016/ https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/2016/omni_hro2_1min_20160101_v01.cdf
# echo "OMNI data downloaded"

echo "Downloading WIND data"

wget --no-clobber --directory-prefix=data/raw/wind/3dp/3dp_elm2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/2016/wi_elm2_3dp_201601{01..07}_v02.cdf
wget --no-clobber --directory-prefix=data/raw/wind/3dp/3dp_pm/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_pm/2016/wi_pm_3dp_201601{01..07}_v05.cdf
wget --no-clobber --directory-prefix=data/raw/wind/mfi/mfi_h2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/wi_h2_mfi_201601{01..07}_v05.cdf

# Download all CDF files and sub-directories from a directory, removing the first two directories from the saved filepath
## In Raapoi terminal: 10.7MB/s

#wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/
#wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/
#wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_pm/
#wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/

# We don't want data before 1995 or after 2022, do delete these directories
# (I think this is easier than specifying it in the wget command)
rm -rf $(find . -type d -name "19[0-8][0-9]" -o -name "199[0-4]")
rm -rf $(find . -type d -name "202[3-9]")

echo "WIND data downloaded"

echo "FINISHED"
