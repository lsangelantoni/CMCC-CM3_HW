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

# input variables are defined in the shell script run_make_hw*.sh
# In this way you can run in parallel several models

ystart=           int(sys.argv[1])
yend=             int(sys.argv[2])

lon_min=          int(sys.argv[3])
lon_max=          int(sys.argv[4])
lat_min=          int(sys.argv[5])
lat_max=          int(sys.argv[6])

tasmax_dir=       str(sys.argv[7])
p90_dir=          str(sys.argv[8])
hfls_dir=         str(sys.argv[9])

cam_lsm_dir=      str(sys.argv[10])
models=           str(sys.argv[11])
f_out=            str(sys.argv[12])


# In[3]:
cam_lsm=xr.open_dataset(f'{cam_lsm_dir}/SST_CMCC-CM3-LT.cam.sst_pcmdi_amip.cntl.1m.1990.nc')
ds_mod1 = xr.open_mfdataset(f'{tasmax_dir}/tasmax_CMCC-CM3_BlueAdapt_HISTORICAL*.nc')
ds_mod1 = ds_mod1.sel(time=ds_mod1['time'].dt.month.isin([6, 7, 8]))
ds_mod2 = xr.open_mfdataset(f'{tasmax_dir}/tasmax_CMCC-CM3_BlueAdapt_SSP585_6hr*.nc').resample(time='D').max()
ds_mod2 = ds_mod2.sel(time=ds_mod2['time'].dt.month.isin([6, 7, 8]))
ds_tasmax=xr.concat([ds_mod1,ds_mod2],dim='time')


# In[4]:
ds_mod1 = xr.open_mfdataset(f'{hfls_dir}/hfls_CMCC-CM3_BlueAdapt_HISTORICAL*.nc')
ds_mod1 = ds_mod1.sel(time=ds_mod1['time'].dt.month.isin([6, 7, 8]))
ds_mod2 = xr.open_mfdataset(f'{hfls_dir}/hfls_CMCC-CM3_BlueAdapt_SSP585_6hr*.nc').resample(time='D').mean() # Differnet daily stat from tasmax!
ds_mod2 = ds_mod2.sel(time=ds_mod2['time'].dt.month.isin([6, 7, 8]))
ds_hfls = xr.concat([ds_mod1,ds_mod2],dim='time')


# In[5]:
ds_p90 = xr.open_dataset(f'{p90_dir}/tasmax_CMCC-CM3_BlueAdapt_HISTORICAL_day_1974-1994.nc')
ds_p90 = ds_p90.sel(time=ds_p90['time'].dt.month.isin([6, 7, 8]))


# In[6]:
mask = cam_lsm.SST[0,:,:]!=0


# In[7]:
#change lon coord to -180,180
if ds_tasmax.lon.min() >= 0:
    with xr.set_options(keep_attrs=True):
        ds_tasmax['lon'] = ((ds_tasmax.lon + 180) % 360) - 180
        ds_tasmax = ds_tasmax.sortby(ds_tasmax.lon)
        ds_hfls['lon'] = ((ds_hfls.lon + 180) % 360) - 180
        ds_hfls = ds_hfls.sortby(ds_hfls.lon)
        ds_p90['lon'] = ((ds_p90.lon + 180) % 360) - 180
        ds_p90 = ds_p90.sortby(ds_p90.lon)
        mask['lon'] = ((mask.lon + 180) % 360) - 180
        mask = mask.sortby(mask.lon)


# In[8]:
ds_tasmax = ds_tasmax.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds_hfls = ds_hfls.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds_p90 = ds_p90.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
mask = mask.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).T


# In[9]:
def compute_heatwave_metrics(
    ds_tasmax,
    ds_hfls,
    ds_p90,
    mask,
    f_out,
    models,
    ystart,
    yend,
    min_duration=3,
):

    """
    Computes heatwave metrics based on input climate data.

    Parameters:
        ds_tasmax (xr dataset): tasmax data.
        ds_hfls (xr dataset): hfls data.
        ds_p90 (xr dataset): tasmax 90th percentile data.
        mask (xr dataarray): cam land sea mask.
        f_out (str): Output file path for storing results.
        models (list): List of model names.
        ystart (int): Start year.
        yend (int): End year.
        min_duration (int): Minimum duration (days) for defining a heatwave event.
    Returns:
        None
    """
   
  
    # Initialize output arrays
    nlat, nlon = ds_tasmax.tasmax.shape[1], ds_tasmax.tasmax.shape[2]
    years = np.arange(ystart, yend)
    tasmax_annual_max=np.full((len(years), nlat, nlon), np.nan)
    hw_metrics = {
        "persistence": np.full((len(years), nlat, nlon), np.nan),
        "index_event_start": np.full((len(years), nlat, nlon), np.nan),
        "index_event_end": np.full((len(years), nlat, nlon), np.nan),
        "mean_tmax": np.full((len(years), nlat, nlon), np.nan),
        "max_tmax": np.full((len(years), nlat, nlon), np.nan),
        "number": np.full((len(years), nlat, nlon), np.nan),
        "hwmi": np.full((len(years), nlat, nlon), np.nan),
        "hfls_deficit": np.full((len(years), nlat, nlon), np.nan),
        "mean_hfls": np.full((len(years), nlat, nlon), np.nan),
    }

    # Iterate over years
    for year_idx, year in enumerate(years):
        print(f"Processing model: {models}, year: {year}")
        
        y_tasmax = ds_tasmax.sel(time=ds_tasmax.time.dt.year == year).tasmax.where(mask == 0, np.nan).values
        y_hfls   = ds_hfls.sel(time=ds_hfls.time.dt.year == year).hfls.where(mask == 0, np.nan).values
        y_p90    = ds_p90.tasmax.where(mask == 0, np.nan).values
                   

        # Iterate over grid points
        for ii in range(nlat):
            for jj in range(nlon):
                if not np.isnan(y_tasmax[0, ii, jj]):
                    array = y_tasmax[:, ii, jj]
                    thresh = y_p90[:, ii, jj]
                    array_hfls = y_hfls[:, ii, jj]
                    tasmax_annual_max = np.nanmax(array)
                    # Compute heatwave events
                    events, indices = [], []
                    i = 0
                    while i < len(array):
                        if array[i] >= thresh[i]:
                            start_idx = i
                            while i < len(array) and array[i] >= thresh[i]:
                                i += 1
                            duration = i - start_idx
                            if duration >= min_duration:
                                events.append(duration)
                                indices.append((start_idx, i - 1))
                        else:
                            i += 1

                    # Compute metrics for heatwave events
                    if events:
                        amplitudes = [
                            np.sum((array[start:end + 1] - np.percentile(array, 25)) /
                                   scipy.stats.iqr(array))
                            for start, end in indices
                        ]
                        max_idx = np.argmax(amplitudes)  # 0 for a single event
                        start, end = indices[max_idx]

                        hw_metrics["persistence"][year_idx, ii, jj] = events[max_idx]
                        hw_metrics["index_event_start"][year_idx, ii, jj] = start
                        hw_metrics["index_event_end"][year_idx, ii, jj] = end
                        hw_metrics["mean_tmax"][year_idx, ii, jj] = np.mean(array[start:end + 1])
                        hw_metrics["max_tmax"][year_idx, ii, jj] = np.max(array[start:end + 1])
                        hw_metrics["number"][year_idx, ii, jj] = len(events)
                        hw_metrics["hwmi"][year_idx, ii, jj] = amplitudes[max_idx]
                        hw_metrics["hfls_deficit"][year_idx, ii, jj] = np.sum(
                            (array_hfls[start:end + 1] - np.percentile(array_hfls, 75)) /
                            scipy.stats.iqr(array_hfls)
                        ) * -1
                        hw_metrics["mean_hfls"][year_idx, ii, jj] = np.mean(array_hfls[start:end + 1])

   
    return(hw_metrics,tasmax_annual_max)

# In[10]  -> run hw metrics:
hw_metrics = compute_heatwave_metrics(
    ds_tasmax,
    ds_hfls,
    ds_p90,
    mask,
    f_out,
    models,
    ystart,
    yend)


# In[11] -> save outputs in a netcdf file:
def save_to_netcdf_with_xarray(
    f_out, 
    lat, 
    lon, 
    time,  
    HW_persistence, 
    HW_HWMI, 
    HW_index_event_start, 
    HW_index_event_end, 
    HW_mean_tmax, 
    HW_max_tmax, 
    HW_number, 
    HW_hfls_deficit, 
    HW_mean_hfls, 
    tasmax_annual_max,
    model_name
):
    # Create an xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "persistence": (
                ("time", "lat", "lon"), HW_persistence,
                {"long_name": "HW persistence", "units": "DAYS"}
            ),
            "HWMI": (
                ("time", "lat", "lon"), HW_HWMI,
                {"long_name": "Heatwave magnitude index daily", "units": "HWMId"}
            ),
            "index_event_start": (
                ("time", "lat", "lon"), HW_index_event_start,
                {"long_name": "HW start time index", "units": "summer day"}
            ),
            "index_event_end": (
                ("time", "lat", "lon"), HW_index_event_end,
                {"long_name": "HW end time index", "units": "summer day"}
            ),
            "mean_tmax": (
                ("time", "lat", "lon"), HW_mean_tmax,
                {"long_name": "HW days mean maximum temperature", "units": "°C"}
            ),
            "max_tmax": (
                ("time", "lat", "lon"), HW_max_tmax,
                {"long_name": "HW days max maximum temperature", "units": "°C"}
            ),
            "number": (
                ("time", "lat", "lon"), HW_number,
                {"long_name": "HW number", "units": "N."}
            ),
            "hfls_deficit": (
                ("time", "lat", "lon"), HW_hfls_deficit,
                {"long_name": "HW hfls deficit", "units": "unitless"}
            ),
            "mean_hfls": (
                ("time", "lat", "lon"), HW_mean_hfls,
                {"long_name": "mean HW hfls", "units": "W/m^2"}
            ),
            "annual_tasmax_max": (
                ("time", "lat", "lon"), txx,
                {"long_name": "annual tmax maximum", "units": "°C"}
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
lat = ds_tasmax.lat
lon = ds_tasmax.lon
time = pd.date_range(start=f'{ystart}', end=f'{yend}', freq="A-JUN")

# Call the function
save_to_netcdf_with_xarray(
    f_out=f'{f_out}',
    lat=lat.values,
    lon=lon.values,
    time=time.values,  # Pass the time array
    HW_persistence=hw_metrics['persistence'],
    HW_HWMI=hw_metrics['hwmi'],
    HW_index_event_start=hw_metrics['index_event_start'],
    HW_index_event_end=hw_metrics['index_event_end'],
    HW_mean_tmax=hw_metrics['mean_tmax'],
    HW_max_tmax=hw_metrics['max_tmax'],
    HW_number=hw_metrics['number'],
    HW_hfls_deficit=hw_metrics['hfls_deficit'],
    HW_mean_hfls=hw_metrics['mean_hfls'],
    txx=annual_tasmax_max,
    model_name=models
)


# In[12] make some tests:
ds=xr.open_dataset(f'{f_out}')

fig, axs = plt.subplots(ncols=1,nrows=1,
                        subplot_kw={'projection': ccrs.PlateCarree()})

P1=axs.pcolormesh(ds_tasmax.lon,ds_tasmax.lat,np.nanmean(ds.mean_tmax,axis=0),transform=ccrs.PlateCarree())
axs.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
axs.add_feature(cfeature.BORDERS.with_scale('50m'))
gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                       color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False      

cb = fig.colorbar(P1, ax=(axs), orientation='vertical',label='°C',aspect=15,shrink=.75,pad=.015)
plt.savefig('test_hws.png')



