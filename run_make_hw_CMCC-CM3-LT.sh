#! /bin/bash

#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J make_hw_CMCC-CM3-LT
#BSUB -P 0564

lon_min=-11
lon_max=40
lat_min=30
lat_max=70

for y in {1975..2014} ; 
do 
	tasmax_dir="/data/cmcc/ls21622/CESM/postprocessed/all/tasmax"
	p90_dir="/data/cmcc/ls21622/CESM/postprocessed/historical/day/tasmax/ANNUAL/p90"
	hfls_dir="/data/cmcc/ls21622/CESM/postprocessed/all/hfls"
	hfss_dir="/data/cmcc/ls21622/CESM/postprocessed/all/hfss"
	cam_lsm_dir="/data/cmcc/ls21622/CESM/CMCC-CM3-LT_cntl_1990_cyc_sst_ymonmean/atm/cntl_1/monthly"
	models='CMCC-CM3-LT'
	f_out="/work/cmcc/ls21622/tmp_hw_metrics/"$models"_heatwave_metrics_day_"$y".nc"
	
	python /users_home/cmcc/ls21622/CMCC-CM3_HW/scripts/python/hw/make_hw_CMCC-CM3-LT.py "$y" "$lon_min" "$lon_max" "$lat_min" "$lat_max" "$tasmax_dir" "$p90_dir" "$hfls_dir" "$hfss_dir" "$cam_lsm_dir" "$models" "$f_out"
done 

