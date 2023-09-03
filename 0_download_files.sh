#!/bin/bash -e

######## FOR BULK DOWNLOADING FROM NASA'S SPACE PHYSICS DATA FACILITY (CDAWEB) ########

# Currently run in an interactive Rāpoi session using tmux
# (can also run locally on a small number of files)

# Get solar cycle (sunspot) data from SIDC (Solar Influences Data Analysis Center)
echo "Downloading sunspot data..."
wget --directory-prefix=data/raw/sunspots https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt
echo "Sunspot data downloaded."

# Download a sequence of files from a specific directory
## (Currently downloading 1 week worth of data: takes about 4min locally, 20s on Google Colab)

echo "Downloading OMNI data"
wget --directory-prefix=data/raw/omni/omni_cdaweb/hro2_1min/2016/ https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/2016/omni_hro2_1min_20160101_v01.cdf
echo "OMNI data downloaded"

echo "Downloading WIND data"

wget --directory-prefix=data/raw/wind/3dp/3dp_elm2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/2016/wi_elm2_3dp_201601{01..07}_v02.cdf
wget --directory-prefix=data/raw/wind/3dp/3dp_plsp/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_plsp/2016/wi_plsp_3dp_201601{01..07}_v02.cdf
wget --directory-prefix=data/raw/wind/mfi/mfi_h2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/wi_h2_mfi_201601{01..07}_v05.cdf

# Download all CDF files and sub-directories from a directory, removing the first two directories from the saved filepath
## In Raapoi terminal: 10.7MB/s

#wget --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/
#wget --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/
#wget --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_plsp/
#wget --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/

echo "WIND data downloaded"

echo "FINISHED"
