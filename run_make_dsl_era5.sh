#! /bin/bash
#BSUB -n 1
#BSUB -R "rusage[mem=5G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J make_dsl_ERA5
#BSUB -P 0564

ystart=1975
yend=2025
lon_min=-11
lon_max=40
lat_min=30
lat_max=70

pr_dir="/data/cmcc/ls21622/ERA5/pr/day"
lsm_file="/data/cmcc/ls21622/ERA5/LSM/ERA5_lsm.nc"
models='ERA5'
f_out="/work/cmcc/ls21622/tmp_dsl_metrics/"$models"_dsl_metrics_day_"$ystart"-"$[yend-1]".nc"

python /users_home/cmcc/ls21622/CMCC-CM3_HW/scripts/python/dsl/make_dsl_ERA5.py "$ystart" "$yend" "$lon_min" "$lon_max" "$lat_min" "$lat_max" "$pr_dir" "$lsm_file" "$models" "$f_out"  

