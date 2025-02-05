#! /bin/bash

#BSUB -n 1
#BSUB -R "rusage[mem=60G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J make_hw_era5
#BSUB -P 0564

y=1975
lon_min=-11
lon_max=40
lat_min=30
lat_max=70

# !! 
# Set a smaller time/spatial domain if testing
# !! 

tasmax_dir="/data/cmcc/ls21622/ERA5/ERA5_land/tmax/tmp"
p90_dir="/data/cmcc/ls21622/ERA5/ERA5_land/tmax/p90/"
hfls_dir="/data/cmcc/ls21622/ERA5/ERA5_land/hfls/tmp"
models='ERA5-land'
f_out="/work/cmcc/ls21622/tmp_hw_metrics/"$models"_heatwave_metrics_day_"$y".nc"

python /users_home/cmcc/ls21622/CMCC-CM3_HW/scripts/python/hw/make_hw_ERA5-land.py "$y" "$lon_min" "$lon_max" "$lat_min" "$lat_max" "$tasmax_dir" "$p90_dir" "$hfls_dir" "$models" "$f_out"

