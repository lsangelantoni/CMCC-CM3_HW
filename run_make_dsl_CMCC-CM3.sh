#! /bin/bash
#BSUB -n 1
#BSUB -R "rusage[mem=5G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J make_dsl_CMCC-CM3
#BSUB -P 0564

ystart=1975
yend=2025
lon_min=-11
lon_max=40
lat_min=30
lat_max=70

pr_dir="/data/cmcc/ls21622/CESM/postprocessed/all/pr"
cam_lsm="/data/cmcc/ls21622/CESM/CMCC-CM3-LT_cntl_1990_cyc_sst_ymonmean/atm/cntl_1/monthly//SST_CMCC-CM3-LT.cam.sst_pcmdi_amip.cntl.1m.1990.nc"
models='CMCC-CM3-LT'
f_out="/work/cmcc/ls21622/tmp_dsl_metrics/"$models"_dsl_metrics_day_"$ystart"-"$[yend-1]".nc"

python /users_home/cmcc/ls21622/CMCC-CM3_HW/scripts/python/dsl/make_dsl.py "$ystart" "$yend" "$lon_min" "$lon_max" "$lat_min" "$lat_max" "$pr_dir" "$cam_lsm" "$models" "$f_out"  

