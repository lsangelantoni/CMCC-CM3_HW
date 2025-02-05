#! /bin/bash

#BSUB -n 1
#BSUB -R "rusage[mem=100G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J make_hw
#BSUB -P 0564

y=1975
lon_min=-11
lon_max=40
lat_min=30
lat_max=70

tasmax_dir="/data/cmcc/ls21622/CESM/postprocessed/all/tasmax"
p90_dir="/data/cmcc/ls21622/CESM/postprocessed/historical/day/tasmax/ANNUAL/p90"
hfls_dir="/data/cmcc/ls21622/CESM/postprocessed/all/hfls"
cam_lsm_dir="/data/cmcc/ls21622/CESM/CMCC-CM3-LT_cntl_1990_cyc_sst_ymonmean/atm/cntl_1/monthly"
models='CMCC-CM3-LT'
f_out="/work/cmcc/ls21622/tmp_hw_metrics/"$models"_heatwave_metrics_day_"$y".nc"

python /users_home/cmcc/ls21622/CMCC-CM3_HW/scripts/python/hw/make_all_days_hw.py "$y" "$lon_min" "$lon_max" "$lat_min" "$lat_max" "$tasmax_dir" "$p90_dir" "$hfls_dir" "$cam_lsm_dir" "$models" "$f_out"

