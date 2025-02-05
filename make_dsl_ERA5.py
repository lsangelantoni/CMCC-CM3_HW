#!/usr/bin/env python
import os
os.environ["ESMFMKFILE"] = "/users_home/cmcc/ls21622/.conda/envs/DEVELOP/lib/esmf.mk"
from netCDF4 import Dataset, num2date
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import pandas as pd
from collections import OrderedDict
import xarray as xr
import scipy
from scipy import special
from scipy.stats import norm
from scipy.stats import ks_2samp as ks_2samp
import datetime as date
import cftime
import dask
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe
import os
import datetime
from cftime import DatetimeGregorian
from scipy.spatial import cKDTree
import seaborn as sns
import dill 
import pymannkendall as mk
import dask.array as da
from dask import delayed, compute
import regionmask 
import sys

class structtype():
    pass
    
ystart=           int(sys.argv[1])
yend=             int(sys.argv[2])

lon_min=          int(sys.argv[3])
lon_max=          int(sys.argv[4])
lat_min=          int(sys.argv[5])
lat_max=          int(sys.argv[6])

pr_dir=           str(sys.argv[7])

lsm_file=         str(sys.argv[8])
models=           str(sys.argv[9])
f_out=            str(sys.argv[10])


lsm=xr.open_dataset(f'{lsm_file}').rename({'var172':'lsm'}).squeeze()
ds_pr = xr.open_mfdataset(f'{pr_dir}/ERA5_pr_day_*').rename({'valid_time':'time','longitude':'lon','latitude':'lat'})
ds_pr = ds_pr.sel(time=ds_pr['time'].dt.month.isin([6, 7, 8]))*1000
ds_pr

# In[6]:
mask = lsm.lsm!=0
mask=mask.sortby('lat')

# In[7]:
#change lon coord to -180,180
if ds_pr.lon.min() >= 0:
    with xr.set_options(keep_attrs=True):
        ds_pr['lon'] = ((ds_pr.lon + 180) % 360) - 180
        ds_pr = ds_pr.sortby(ds_pr.lon)
        mask['lon'] = ((mask.lon + 180) % 360) - 180
        mask = mask.sortby(mask.lon)




ds_pr = ds_pr.sortby(ds_pr.lat)
ds_pr = ds_pr.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
mask = mask.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).T


# Initialize output arrays
nlat, nlon = ds_pr.tp.shape[1], ds_pr.tp.shape[2]
years = np.arange(ystart, yend)
pr_int=np.full((len(years), nlat, nlon), np.nan)

class structtype():
    pass


def count_dsl(array,
              thresh,
              Count,
              i,
              cval,
              dummy_index,
              pacchetti,
              index_event_start,
              index_event_end,
              index_event,
              index_events):
                

    while i < len(array)-1 : 
        
        
        if array[i] < thresh : 
            
            i = i + 1;
                
            if i >= len(array) : 
                    
                break
                                    
            c = 1;
                
            while array[i] < thresh : # Start counting (c) time steps
                    
                c = c + 1
                i = i + 1
                    
                if i >= len(array) :
                        
                    break
                        
                    
            if c >= cval : # Start counting (Count) when "c" is above 2 consecutive days
    
                    Count = Count + 1
    
                    pacchetti[Count] = c
                    
                    dummy_index[Count] = i
                    index_event_end[Count]   = int(dummy_index[Count]) #-1 
                    index_event_start[Count] = int(dummy_index[Count] - pacchetti[Count]) 
                    index_event[Count] = np.array(np.arange(int(index_event_start[Count]), int(index_event_end[Count]),1))
                    
                    # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
                    dummy_len=len(np.array(np.arange(int(index_event_start[Count]), int(index_event_end[Count]),1)))
                    index_events[Count,range(0,dummy_len)] = np.array(np.arange(int(index_event_start[Count]), int(index_event_end[Count]),1))
                    # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
                    
        else :
            i = i + 1

    return pacchetti



cc = -1

out_index_event_start = np.full(([len(range(ystart,yend)),ds_pr.tp.shape[1],ds_pr.tp.shape[2]]),np.nan)
out_index_event_end   = np.full(([len(range(ystart,yend)),ds_pr.tp.shape[1],ds_pr.tp.shape[2]]),np.nan)
out_mean              = np.full(([len(range(ystart,yend)),ds_pr.tp.shape[1],ds_pr.tp.shape[2]]),np.nan)
out_max               = np.full(([len(range(ystart,yend)),ds_pr.tp.shape[1],ds_pr.tp.shape[2]]),np.nan)
out_number            = np.full(([len(range(ystart,yend)),ds_pr.tp.shape[1],ds_pr.tp.shape[2]]),np.nan)

# Iterate over years
for year_idx, year in enumerate(years):
    cc=cc+1
    print(f"Processing model: {models}, year: {year}")
    pr_y = ds_pr.sel(time=ds_pr.time.dt.year == year).tp.where(mask != 0, np.nan).values
    
    for ii in range(nlat):
    
        for jj in range(nlon):
        
            if not np.isnan(pr_y[0, ii, jj]):
    
                pacchetti         = np.empty(92)*np.nan
                dummy_index       = np.empty(92)*np.nan
                index_event_start = np.empty(92)*np.nan
                index_event_end   = np.empty(92)*np.nan
                index_events      = np.empty([92,92]) * np.nan # It considers all the DSLs in one year
                index_event = [ structtype() for i in range(92) ]
                array  = pr_y[:,ii,jj]
                pr_int[year_idx,ii,jj] = np.nanmean(array[array>0]) 
                thresh = 1
                Count = -1
                i = 0
                cval = 2 # Attention here, the value from which we start counting DSL. HWS was three cons. days here two.
                pacchetti = count_dsl(
                    array,
                    thresh,
                    Count,
                    i,
                    cval,
                    dummy_index,
                    pacchetti,
                    index_event_start,
                    index_event_end,
                    index_event,
                    index_events)
                    
                index_events = index_events[~np.isnan(index_events)]
                index_events = [ int(x) for x in index_events ]


                if pd.isna(pacchetti[0]) != True :
                    pacchetti = pacchetti[~np.isnan(pacchetti)]
                    index_event_max = (index_event[(np.argmax(pacchetti))])
                    out_index_event_start[cc,ii,jj] = np.min(index_event_max)
                    out_index_event_end[cc,ii,jj]   = np.max(index_event_max)
                    out_mean[cc,ii,jj] = np.round(np.nanmean(pacchetti))
                    out_max[cc,ii,jj]  = np.round(np.nanmax(pacchetti))
                    out_number[cc,ii,jj] = len(pacchetti)



def save_to_netcdf_with_xarray(
    f_out, 
    lat, 
    lon, 
    time,  
    dsl_mean,
    dsl_max,
    dsl_number,
    INT, 
    model_name
):
    # Create an xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "dsl_mean": (
                ("time", "lat", "lon"), dsl_mean.astype(float),
                {"long_name": "annnual mean dry spell length", "units": "days"}
            ),
            "dsl_max": (
                ("time", "lat", "lon"), dsl_max.astype(float),
                {"long_name": "annual maximum dry spell length", "units": "days"}
            ),
            "dsl_number": (
                ("time", "lat", "lon"), dsl_number,
                {"long_name": "annual number of dry spells", "units": "n."}
            ),
            "INT": (
                ("time", "lat", "lon"), INT,
                {"long_name": "mean annual precipitation intensity during wet days (>1mm/day)", "units": "mm/day"}
            ),
        },
        coords={
            "time": ("time", time, {"long_name": "time"}),
            "lat": ("lat", lat, {"long_name": "latitude", "units": "degree_north"}),
            "lon": ("lon", lon, {"long_name": "longitude", "units": "degree_east"}),
        },
        attrs={"model_name": model_name}  # Global attribute for the model name
    )

    ds.to_netcdf(f_out, format="NETCDF4")



# Example data dimensions
lat = ds_pr.lat
lon = ds_pr.lon
time = pd.date_range(start=f'{ystart}', end=f'{yend}', freq="A-JUN")

# Call the function
save_to_netcdf_with_xarray(
    f_out=f'{f_out}',
    lat=lat.values,
    lon=lon.values,
    time=time.values,  # Pass the time array
    dsl_mean=out_mean,
    dsl_max=out_max,
    dsl_number=out_number,
    INT=pr_int,
    model_name=models
)


dataset = Dataset(f'{f_out}', mode='r')
lon=dataset['lon']
lat=dataset['lat']
var=dataset['dsl_mean']



fig, axs = plt.subplots(ncols=1,nrows=1,
                        subplot_kw={'projection': ccrs.PlateCarree()})

P1=axs.pcolormesh(lon,lat,np.nanmean(var,axis=0),transform=ccrs.PlateCarree())
axs.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
axs.add_feature(cfeature.BORDERS.with_scale('50m'))
#axs.set_extent([-100,80,20,70])
gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                       color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False      

cb = fig.colorbar(P1, ax=(axs), orientation='vertical',label='mm/day',aspect=15,shrink=.75,pad=.015)

plt.savefig('test_dsl.png')



