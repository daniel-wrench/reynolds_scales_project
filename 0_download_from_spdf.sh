######## FOR BULK DOWNLOADING FROM NASA'S SPACE PHYSICS DATA FACILITY (CDAWEB) ########

# Use the following to download a sequence of files from a single directory
# (Currently downloading 2 months worth of data)
wget --directory-prefix=data/raw/omni_hro2_1min/ https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/2016/omni_hro2_1min_2016{01..02}01_v01.cdf
wget --directory-prefix=data/raw/wind/3dp/3dp_elm2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/2016/wi_elm2_3dp_2016{01..02}{01..31}_v02.cdf
wget --directory-prefix=data/raw/wind/3dp/3dp_plsp/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_plsp/2016/wi_plsp_3dp_2016{01..02}{01..31}_v02.cdf
wget --directory-prefix=data/raw/wind/mfi/mfi_h2/2016/ https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/wi_h2_mfi_2016{01..02}{01..31}_v05.cdf

# Use the following to download all CDF files and sub-directories from a directory, removing the first two directories from the saved filepath
#wget --directory-prefix=data/raw/ --recursive -np -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/