#! /bin/bash
#BSUB -n 1
#BSUB -R "rusage[mem=15G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J make_hw
#BSUB -P 0564

ystart=1975
yend=2025
lon_min=-100
lon_max=100
lat_min=20
lat_max=80

tx_dir="/data/cmcc/ls21622/CESM/postprocessed/all/tasmax"
p90_dir="/data/cmcc/ls21622/CESM/postprocessed/historical/day/tasmax/ANNUAL/p90"
hfls_dir="/data/cmcc/ls21622/CESM/postprocessed/all/hfls"
cam_lsm_dir="/data/cmcc/ls21622/CESM/CMCC-CM3-LT_cntl_1990_cyc_sst_ymonmean/atm/cntl_1/monthly"
models='CMCC-CM3-LT'
f_out="/work/cmcc/ls21622/tmp_hw_metrics/"$models"_heatwave_metrics_day_"$ystart"-"$[yend-1]".nc"

python /users_home/cmcc/ls21622/CMCC-CM3_HW/scripts/python/make_hw.py "$ystart" "$yend" "$lon_min" "$lon_max" "$lat_min" "$lat_max" "$tx_dir" "$p90_dir" "$hfls_dir" "$cam_lsm_dir" "$models" "$f_out" > log.py 

