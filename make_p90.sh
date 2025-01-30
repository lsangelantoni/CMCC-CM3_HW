#!/bin/bash

#BSUB -n 1
#BSUB -R "rusage[mem=10G]"
#BSUB -q s_long
#BSUB -o logs/
#BSUB -e logs/
#BSUB -J DO_90P
#BSUB -P 0564

module load intel-2021.6.0/cdo-threadsafe/2.1.1-lyjsw
module load intel-2021.6.0/nco
module load intel-2021.6.0/netcdf-c/4.9.0-cjqig

# This the second version of the script to derive 90p along a 31-day time window. 

 minlon=-100
 maxlon=100
 minlat=20
 maxlat=90

# ---------------------------------------------------------------------------------  
  models=CMCC-CM3
  var=tasmax
  period="historical" #1975-2024 historical + SSP585 segments. 
  TW=31 #Time window size for computing running 90p

  export REMAP_EXTRAPOLATE=off
# --------------------------------------------------------------------------------


# + + + + + + + + + + + + + + + + + + +
echo " COMPUTE 90P" 
# + + + + + + + + + + + + + + + + + + +
for m in {0..0} ; do

	root_dir="/data/cmcc/ls21622/CESM/postprocessed/historical/day/tasmax/ANNUAL"
	work_dir="/work/cmcc/ls21622/temperature_extremes_CMCC-CM3-LT/tmp"
	out_dir=$work_dir"/p90"
	out_file="tasmax_CMCC-CM3_BlueAdapt_HISTORICAL_day_1974-1994.nc"
	mkdir -p $out_dir
	cd $work_dir
        ln -sf $root_dir/*.nc .  	

        cdo mergetime *$models*.nc tmp.nc; #ncap2 -s "lon=lon-360*(lon>180)" tmp.nc tmp2.nc 
	cdo sellonlatbox,$minlon,$maxlon,$minlat,$maxlat -selyear,1975/1994 -selmonth,5/9 tmp.nc ifile.nc
	
	cdo -O ydrunmin,$TW ifile.nc minfile
	cdo -O ydrunmax,$TW ifile.nc maxfile
        cdo -O ydrunpctl,90,$TW ifile.nc minfile maxfile $out_dir/$out_file 
        rm minfile maxfile ifile.nc tmp.nc 
done





