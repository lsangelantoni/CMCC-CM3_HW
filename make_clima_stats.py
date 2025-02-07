#!/usr/bin/env python
# coding: utf-8

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
import glob


ystart=1975;yend=1994
lon_min=-11
lon_max=40
lat_min=30
lat_max=70


### CMCC-CM3-LT
tasmax_dir="/data/cmcc/ls21622/CESM/postprocessed/all/tasmax"
def process_file(file_path, var_name, ystart,yend, resample=False):
    ds = xr.open_dataset(file_path)
    ds = ds.sel(time=ds['time'].dt.month.isin([6, 7, 8]))
    #change lon coord to -180,180
    if ds.lon.min() >= 0:
        with xr.set_options(keep_attrs=True):
            ds['lon'] = ((ds.lon + 180) % 360) - 180
            ds = ds.sortby(ds.lon)
    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return ds

tasmax_files=[]
tasmax_files.extend(glob.glob(f'{tasmax_dir}/tasmax_CMCC-CM3_BlueAdapt_HISTORICAL_day_19*.nc'))
ds = [process_file(f, 'tasmax', ystart,yend) for f in tasmax_files]
ds = xr.concat(ds, dim='time')
ds = ds.sortby(ds.time).drop_dims('bnds')
ds = ds.sel(time=ds['time'].dt.year.isin(range(ystart,yend+1)))
tasmax_clima_p25 = np.percentile(ds.tasmax,25,axis=0)
tasmax_clima_iqr = scipy.stats.iqr(ds.tasmax,rng=(25,75),axis=0) 

hfls_dir="/data/cmcc/ls21622/CESM/postprocessed/all/hfls"
hfls_files=[]
hfls_files.extend(glob.glob(f'{hfls_dir}/hfls_CMCC-CM3_BlueAdapt_HISTORICAL_day_19*.nc'))
ds = [process_file(f, 'hfls', ystart,yend) for f in hfls_files]
ds = xr.concat(ds, dim='time')
ds = ds.sortby(ds.time).drop_dims('bnds')
ds = ds.sel(time=ds['time'].dt.year.isin(range(ystart,yend+1)))
hfls_clima_p75 = np.percentile(ds.hfls,75,axis=0)
hfls_clima_iqr = scipy.stats.iqr(ds.hfls,rng=(25,75),axis=0) 


ds_out = xr.Dataset(
    {
        
        "tasmax_clima_p25": (["lat", "lon"], tasmax_clima_p25, 
                           {"long_name": "25th percentile of tasmax", "units": "Celsius degree"}),
        "tasmax_clima_iqr": (["lat", "lon"], tasmax_clima_iqr, 
                           {"long_name": "Interquartile range of tasmax", "units": "Celsius degree"}),
        "hfls_clima_p75": (["lat", "lon"], hfls_clima_p75, 
                           {"long_name": "75th percentile of hfls", "units": "W/m^2"}),
        "hfls_clima_iqr": (["lat", "lon"], hfls_clima_iqr, 
                           {"long_name": "Interquartile range of hfls", "units": "W/m^2"}),
    },
    coords={
        "lat": ds.lat,  # Assuming ds has latitude coordinates
        "lon": ds.lon,  # Assuming ds has longitude coordinates
    },
)

# Define global attributes
ds_out.attrs["description"] = f'tasmax and Latent Heat flux statistics derived from historical data to be called for HW metrics computation. Period:{ystart}-{yend}'
ds_out.to_netcdf("/work/cmcc/ls21622/tmp_hw_metrics/CMCC-CM3-LT_clima_stats.nc", format="NETCDF4")


# # #  # # #  # # # 


### ERA5-land 
tasmax_dir="/data/cmcc/ls21622/ERA5/ERA5_land/tmax"
def process_file(file_path, var_name, ystart,yend, resample=False):
    ds = xr.open_dataset(file_path).rename({'valid_time':'time','longitude':'lon','latitude':'lat'}) 
    return ds

tasmax_files=[]
tasmax_files.extend(glob.glob(f'{tasmax_dir}/ERA5-land_tmax_day_19[7-9][0-9]_0[6-8]*.nc'))
ds = [process_file(f, 't2m', ystart,yend) for f in tasmax_files]
ds = xr.concat(ds, dim='time')
ds = ds.sortby(ds.time).rename({'t2m':'tasmax'})
ds = ds.sortby(ds.lat)
ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds = ds.sel(time=ds['time'].dt.year.isin(range(ystart,yend+1)))
ds_tasmax=ds
tasmax_clima_p25 = np.percentile(ds.tasmax,25,axis=0)
tasmax_clima_iqr = scipy.stats.iqr(ds.tasmax,rng=(25,75),axis=0) 



### ERA5-land 
tasmax_dir="/data/cmcc/ls21622/ERA5/ERA5_land/tmax"
def process_file(file_path, var_name, ystart,yend, resample=False):
    ds = xr.open_dataset(file_path).rename({'valid_time':'time','longitude':'lon','latitude':'lat'}) 
    return ds

tasmax_files=[]
tasmax_files.extend(glob.glob(f'{tasmax_dir}/ERA5-land_tmax_day_19[7-9][0-9]_0[6-8]*.nc'))
ds = [process_file(f, 't2m', ystart,yend) for f in tasmax_files]
ds = xr.concat(ds, dim='time')
ds = ds.sortby(ds.time).rename({'t2m':'tasmax'})
ds = ds.sortby(ds.lat)
ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds = ds.sel(time=ds['time'].dt.year.isin(range(ystart,yend+1)))
ds_tasmax=ds
tasmax_clima_p25 = np.percentile(ds.tasmax,25,axis=0)
tasmax_clima_iqr = scipy.stats.iqr(ds.tasmax,rng=(25,75),axis=0) 

def process_file(file_path, var_name, resample=False):
    ds = xr.open_dataset(file_path)
    return ds

hfls_dir="/data/cmcc/ls21622/ERA5/ERA5_land/hfls"
hfls_files=[]
hfls_files.extend(glob.glob(f'{hfls_dir}/ERA5-land_hfls_day_19[7-9][0-9]_0[6-8]*.nc'))
ds = [process_file(f, 'hfls') for f in hfls_files]
ds = xr.concat(ds, dim='time')
ds = ds.sortby(ds.time)
ds = ds.resample(time='D').mean()
ds = ds.sel(time=ds['time'].dt.month.isin([6, 7, 8]))
ds = ds.sortby(ds.lat)
ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

method = 'bilinear'
def regrid(ds,ds_out,method):
    regridder = xe.Regridder(ds,ds_out, method = method)
    ds = regridder(ds,ds_out,method)
    return ds

ds_regrid = regrid(ds,ds_tasmax,method)
ds = ds_regrid

hfls_clima_p75 = np.percentile(ds.hfls,75,axis=0)
hfls_clima_iqr = scipy.stats.iqr(ds.hfls,rng=(25,75),axis=0) 

ds_out = xr.Dataset(
    {
        
        "tasmax_clima_p25": (["lat", "lon"], tasmax_clima_p25, 
                           {"long_name": "25th percentile of tasmax", "units": "Celsius degree"}),
        "tasmax_clima_iqr": (["lat", "lon"], tasmax_clima_iqr, 
                           {"long_name": "Interquartile range of tasmax", "units": "Celsius degree"}),
        "hfls_clima_p75": (["lat", "lon"], hfls_clima_p75, 
                           {"long_name": "75th percentile of hfls", "units": "W/m^2"}),
        "hfls_clima_iqr": (["lat", "lon"], hfls_clima_iqr, 
                           {"long_name": "Interquartile range of hfls", "units": "W/m^2"}),
    },
    coords={
        "lat": ds.lat,  # Assuming ds has latitude coordinates
        "lon": ds.lon,  # Assuming ds has longitude coordinates
    },
)

# Define global attributes
ds_out.attrs["description"] = f'tasmax and Latent Heat flux statistics derived from historical data to be called for HW metrics computation. Period:{ystart}-{yend}'
ds_out.to_netcdf("/work/cmcc/ls21622/tmp_hw_metrics/ERA5-land_clima_stats.nc", format="NETCDF4")


