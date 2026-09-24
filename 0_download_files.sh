#!/bin/bash -e

######## FOR BULK DOWNLOADING FROM NASA'S SPACE PHYSICS DATA FACILITY (CDAWEB) ########

# Currently run in an interactive Rāpoi session using tmux
# (can also run locally on a small number of files)

# This script downloads data files from NASA's Space Physics Data Facility (CDAWeb) and the Solar Influences Data Analysis Center (SIDC). It checks for the presence of either `wget` or `curl` to perform the downloads, and creates necessary directories if they do not exist.

download_file() {
  local url="$1"
  local dest_dir="$2"
  local filename="$(basename "$url")"

  mkdir -p "$dest_dir"

  if command -v wget >/dev/null 2>&1; then
    wget --no-clobber --directory-prefix="$dest_dir" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$dest_dir/$filename" "$url"
  else
    echo "Error: neither wget nor curl is installed. Install one of them and rerun this script." >&2
    exit 1
  fi
}

# Get solar cycle (sunspot) data from SIDC (Solar Influences Data Analysis Center)
# echo "Downloading sunspot data..."
# download_file "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt" "data/raw/sunspots"
# echo "Sunspot data downloaded."

# Download a sequence of files from a specific directory
## (Currently downloading 1 week worth of data: takes about 4min locally, 20s on Google Colab)

echo "Downloading OMNI data"
download_file "https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/2016/omni_hro2_1min_20160101_v01.cdf" "data/raw/omni/omni_cdaweb/hro2_1min/2016/"
echo "OMNI data downloaded"

echo "Downloading WIND data"

for d in 01 02 03 04 05 06 07; do
  download_file "https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/2016/wi_elm2_3dp_201601${d}_v02.cdf" "data/raw/wind/3dp/3dp_elm2/2016/"
done

for d in 01 02 03 04 05 06 07; do
  download_file "https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_pm/2016/wi_pm_3dp_201601${d}_v05.cdf" "data/raw/wind/3dp/3dp_pm/2016/"
done

for d in 01 02 03 04 05 06 07; do
  download_file "https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/wi_h2_mfi_201601${d}_v05.cdf" "data/raw/wind/mfi/mfi_h2/2016/"
done

echo "WIND data downloaded"

# Download ALL CDF files and sub-directories from a directory, removing the first two directories from the saved filepath
## In Raapoi terminal: 10.7MB/s
## These examples are shown in a portable form so they work even when `wget` is unavailable on Windows.
# if command -v wget >/dev/null 2>&1; then
#   wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/
#   wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/
#   wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_pm/
#   wget --no-clobber --directory-prefix=data/raw/ --recursive -np -nv -nH --cut-dirs=2 --accept cdf https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/
# elif command -v curl >/dev/null 2>&1; then
#   mkdir -p data/raw/omni && curl -L --fail --recursive --output-dir data/raw/omni --url https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/
#   mkdir -p data/raw/wind/3dp/3dp_elm2 && curl -L --fail --recursive --output-dir data/raw/wind/3dp/3dp_elm2 --url https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_elm2/
#   mkdir -p data/raw/wind/3dp/3dp_pm && curl -L --fail --recursive --output-dir data/raw/wind/3dp/3dp_pm --url https://spdf.gsfc.nasa.gov/pub/data/wind/3dp/3dp_pm/
#   mkdir -p data/raw/wind/mfi/mfi_h2 && curl -L --fail --recursive --output-dir data/raw/wind/mfi/mfi_h2 --url https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/
# fi

# echo "WIND data downloaded"

echo "FINISHED"
