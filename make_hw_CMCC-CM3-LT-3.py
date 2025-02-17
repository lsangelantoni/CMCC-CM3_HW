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

# Call files with tasmax and hfls statistics for definning HWMId and evaporative deficit. It is already compliant to the considered spatial domain.  
root_clima_stats='/work/cmcc/ls21622/tmp_hw_metrics'
ds_stats=xr.open_dataset(f'{root_clima_stats}/CMCC-CM3-LT_clima_stats.nc') 
# - - - - - - - - - - - - - - - - -


# From sell :                      
y=                int(sys.argv[1])

lon_min=          int(sys.argv[2])
lon_max=          int(sys.argv[3])
lat_min=          int(sys.argv[4])
lat_max=          int(sys.argv[5])

tasmax_dir=       str(sys.argv[6])
p90_dir=          str(sys.argv[7])
hfls_dir=         str(sys.argv[8])
hfss_dir=         str(sys.argv[9])

cam_lsm_dir=      str(sys.argv[10])
models=           str(sys.argv[11])
f_out=            str(sys.argv[12])
# - - - - - - - - - - - - - - - - -



# Set and load land-sea mask 
cam_lsm=xr.open_dataset(f'{cam_lsm_dir}/SST_CMCC-CM3-LT.cam.sst_pcmdi_amip.cntl.1m.1990.nc')
mask = cam_lsm.SST[0,:,:]!=0
# - - - - - - - - - - - - - - - - -


# # # Process tasmax 
# historical
if y <= 2014 : 
    ds = xr.open_dataset(f'{tasmax_dir}/tasmax_CMCC-CM3_BlueAdapt_HISTORICAL_day_{y}0101-{y}1231.nc',engine="netcdf4")
    ds = ds.sel(time=ds.time.dt.month.isin(range(6, 9)))
    ds_tasmax = ds.drop_dims('bnds')
    

# ssp585
def process_file(file_path, var_name, y, resample=False):
    ds = xr.open_dataset(file_path)
       
   # Resample to daily resolution (if required)
    ds = ds.resample(time='D').max('time')
    
    return ds

tasmax_files = []
if y > 2014 : 
    tasmax_files.extend(glob.glob(f'{tasmax_dir}/tasmax_CMCC-CM3_BlueAdapt_SSP585_6hr_{y}-0[6-8].nc'))
    tasmax_datasets_2 = [process_file(f, 'tasmax', y) for f in tasmax_files]
    ds_tasmax_2 = xr.concat(tasmax_datasets_2, dim='time')
    ds_tasmax_2=ds_tasmax_2.sortby(ds_tasmax_2.time)
    ds_tasmax = ds_tasmax_2
# - - - - - - - - - - - - - - - - -

# # # hfls 
# historical
if y <= 2014 : 
    ds = xr.open_dataset(f'{hfls_dir}/hfls_CMCC-CM3_BlueAdapt_HISTORICAL_day_{y}0101-{y}1231.nc',engine="netcdf4")
    ds = ds.sel(time=ds.time.dt.month.isin(range(6, 9)))
    ds_hfls = ds.drop_dims('bnds')

# ssp585
def process_file(file_path, var_name, y, resample=False):
    ds = xr.open_dataset(file_path)
       
   # Resample to daily resolution (if required)
    ds = ds.resample(time='D').mean('time')
    
    return ds

hfls_files = []
if y > 2014 : 
    hfls_files.extend(glob.glob(f'{hfls_dir}/hfls_CMCC-CM3_BlueAdapt_SSP585_6hr_{y}-0[6-8].nc'))
    hfls_datasets_2 = [process_file(f, 'hfls', y) for f in hfls_files]
    ds_hfls_2 = xr.concat(hfls_datasets_2, dim='time')
    ds_hfls_2=ds_hfls_2.sortby(ds_hfls_2.time)
    ds_hfls = ds_hfls_2
# - - - - - - - - - - - - - - - - -



# # # hfss 
def process_file(file_path, var_name, y, resample=False):
    ds = xr.open_dataset(file_path)
    ds = ds.resample(time='D').mean('time')
    
    return ds

# historical
hfss_files = []
if y <= 2014 : 
    hfss_files.extend(glob.glob(f'{hfss_dir}/hfss_CMCC-CM3_BlueAdapt_HISTORICAL_day_{y}-0[6-8].nc'))
    hfss_datasets = [process_file(f, 'hfss', y,resample=True) for f in hfss_files]
    ds_hfss = xr.concat(hfss_datasets, dim='time')
    ds_hfss=ds_hfss.sortby(ds_hfss.time)

# ssp585
hfss_files = []
if y > 2014 : 
    hfss_files.extend(glob.glob(f'{hfss_dir}/hfss_CMCC-CM3_BlueAdapt_SSP585_6hr_{y}-0[6-8].nc'))
    hfss_datasets = [process_file(f, 'hfss', y,resample=True) for f in hfss_files]
    ds_hfss = xr.concat(hfss_datasets, dim='time')
    ds_hfss=ds_hfss.sortby(ds_hfss.time)
# - - - - - - - - - - - - - - - - -


# Call for tasmax p90:
ds_p90 = xr.open_dataset(f'{p90_dir}/tasmax_CMCC-CM3_BlueAdapt_HISTORICAL_day_1974-1994.nc')
ds_p90 = ds_p90.sel(time=ds_p90['time'].dt.month.isin([6, 7, 8]))
# - - - - - - - - - - - - - - - - -


# Change lon coord to -180,180
if ds_tasmax.lon.min() >= 0:
    with xr.set_options(keep_attrs=True):
        ds_tasmax['lon'] = ((ds_tasmax.lon + 180) % 360) - 180
        ds_tasmax = ds_tasmax.sortby(ds_tasmax.lon)
        ds_hfls['lon'] = ((ds_hfls.lon + 180) % 360) - 180
        ds_hfls = ds_hfls.sortby(ds_hfls.lon)
        ds_hfss['lon'] = ((ds_hfss.lon + 180) % 360) - 180
        ds_hfss = ds_hfss.sortby(ds_hfss.lon)
        ds_p90['lon'] = ((ds_p90.lon + 180) % 360) - 180
        ds_p90 = ds_p90.sortby(ds_p90.lon)
        mask['lon'] = ((mask.lon + 180) % 360) - 180
        mask = mask.sortby(mask.lon)
# - - - - - - - - - - - - - - - - -


# Crop domain
ds_tasmax = ds_tasmax.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds_hfls = ds_hfls.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds_hfss = ds_hfss.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds_p90 = ds_p90.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
mask = mask.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).T
# - - - - - - - - - - - - - - - - -



# inputs for HW detection function
nlat,nlon = np.shape(ds_tasmax.tasmax)[1], np.shape(ds_tasmax.tasmax)[2]
np_tasmax = np.array(ds_tasmax.tasmax.where(mask == 0, np.nan))
np_hfls = np.array(ds_hfls.hfls.where(mask == 0, np.nan))
np_hfss = np.array(ds_hfss.hfss.where(mask == 0, np.nan))
np_90p = np.array(ds_p90.tasmax.where(mask == 0, np.nan))

np_p25 = np.array(ds_stats.tasmax_clima_p25.where(mask == 0, np.nan))
np_iqr_tasmax = np.array(ds_stats.tasmax_clima_iqr.where(mask == 0, np.nan))
np_p75 = np.array(ds_stats.hfls_clima_p75.where(mask == 0, np.nan))
np_iqr_hfls = np.array(ds_stats.hfls_clima_iqr.where(mask == 0, np.nan))
# - - - - - - - - - - - - - - - - -


# Function to compute metrics for a single grid point
@delayed
def compute_metrics(tasmax_series, hfls_series, hfss_series, p90_thresh, p25, iqr_tasmax, p75, iqr_hfls,min_duration=3):
    results = np.full(8, np.nan)  # Store all metrics
    
    # Heatwave detection
    events, indices = [], []
    i = 0
    while i < len(tasmax_series)-1:
        if tasmax_series[i] >= p90_thresh[i]:
            start_idx = i
            while i + 1 < len(tasmax_series) and tasmax_series[i + 1] >= p90_thresh[i + 1]:
                i += 1
            duration = i - start_idx + 1
            if duration >= min_duration:
                events.append(duration)
                indices.append((start_idx, i))
        i += 1  # Ensure i always increments

    # Metrics Calculation
    if events:
        hw_days = np.concatenate([tasmax_series[start:end + 1] for start, end in indices])
        hfls_days = np.concatenate([hfls_series[start:end + 1] for start, end in indices])
        hfss_days = np.concatenate([hfss_series[start:end + 1] for start, end in indices])

        results[0] = np.mean(events)                        # Persistence
        results[1] = np.mean(hw_days)                      # Mean Tmax
        results[2] = np.max(hw_days)                       # Max Tmax
        results[3] = len(events)                           # Number of events
        results[4] = np.sum((hw_days - p25) / iqr_tasmax) / len(events)  # HWMI
        results[5] = -1 * np.sum((hfls_days - p75) / iqr_hfls) / len(events)  # HFLS deficit
        results[6] = np.mean(hfls_days)                    # Mean HFLS
        results[7] = np.mean(hfls_days)/(np.mean(hfls_days)+np.mean(hfss_days))  # evaporative fraction hfls/(hfls+hfss)
    return results

# Apply function to all grid points
tasks = []
for ii in range(ds_tasmax.dims['lat']):
    for jj in range(ds_tasmax.dims['lon']):
        # Check if the entire grid cell is not NaN
        if np.isnan(np_tasmax[0, ii, jj]):
            # Skip the task for this grid point, as it has NaN values
            tasks.append(None)  # You can append None or something else to represent NaN
        else:
            tasmax_series = np_tasmax[:, ii, jj]
            hfls_series = np_hfls[:, ii, jj]
            hfss_series = np_hfss[:, ii, jj]
            p90_thresh = np_90p[:,ii, jj]
            p25 = np_p25[ii, jj]
            iqr_tasmax = np_iqr_tasmax[ii, jj]
            p75 = np_p75[ii, jj]
            iqr_hfls = np_iqr_hfls[ii, jj]

            # Only compute metrics if no NaN is encountered
            tasks.append(compute_metrics(tasmax_series, hfls_series, hfss_series, p90_thresh, p25, iqr_tasmax, p75, iqr_hfls, min_duration=3))

# Filter out None tasks (for NaN grid cells)
tasks = [task for task in tasks if task is not None]

# Compute all tasks
results = da.compute(*tasks)

# Convert results to numpy array and reshape
results = np.array(results)

# Initialize the results array with NaNs
final_results = np.full((8, ds_tasmax.dims['lat'], ds_tasmax.dims['lon']), np.nan)

# Fill in the valid grid points
valid_index = 0
for ii in range(ds_tasmax.dims['lat']):
    for jj in range(ds_tasmax.dims['lon']):
        if not np.isnan(np_tasmax[0, ii, jj]):
            # Place the results into the correct grid point
            final_results[:, ii, jj] = results[valid_index]
            valid_index += 1
# - - - - - - - - - - - - - - - - -



# Final_results contains the HW metrics on the lat-lon grid, preserving NaNs
hw_metrics = xr.DataArray(
    final_results,
    dims=["metric", "lat", "lon"],
    coords={
        "metric": ["Persistence", "Mean Tmax", "Max Tmax", "Number of events", "HWMI", "HFLS deficit", "Mean HFLS","mean evaporative fraction"],
        "lat": ds_tasmax.coords["lat"],
        "lon": ds_tasmax.coords["lon"]
    },
    name="heatwave_metrics"
)
# - - - - - - - - - - - - - - - - -


# Save NetCDF
def save_hw_metrics_to_netcdf(
    f_out, 
    lat, 
    lon, 
    time,
    HW_persistence, 
    HW_mean_tmax, 
    HW_max_tmax, 
    HW_number, 
    HW_HWMI, 
    HW_hfls_deficit, 
    HW_mean_hfls,
    HW_mean_evaporative_fraction,
    model_name
):
    
    

    time = pd.to_datetime([f'{y}-06-01'])  # Ensure it's a 1D array
    ds = xr.Dataset(
        data_vars={
            "persistence": (
                ("lat", "lon"), HW_persistence,
                {"long_name": "HW persistence", "units": "DAYS"}
            ),
            "mean_tmax": (
                ("lat", "lon"), HW_mean_tmax,
                {"long_name": "HW days mean maximum temperature", "units": "°C"}
            ),
            "max_tmax": (
                ("lat", "lon"), HW_max_tmax,
                {"long_name": "HW days max maximum temperature", "units": "°C"}
            ),
            "number": (
                ("lat", "lon"), HW_number,
                {"long_name": "HW number", "units": "N."}
            ),
            "HWMI": (
                ("lat", "lon"), HW_HWMI,
                {"long_name": "Heatwave magnitude index daily", "units": "HWMId"}
            ),
            "hfls_deficit": (
                ("lat", "lon"), HW_hfls_deficit,
                {"long_name": "HW hfls deficit", "units": "unitless"}
            ),
            "mean_hfls": (
                ("lat", "lon"), HW_mean_hfls,
                {"long_name": "mean HW hfls", "units": "W/m^2"}
            ),
            "mean_evaporative_fraction": (
                ("lat", "lon"), HW_mean_evaporative_fraction,
                {"long_name": "mean HW evaporative fraction", "units": "ratio"}
            ),
        },
        coords={
            "lat": ("lat", lat, {"long_name": "latitude", "units": "degree_north"}),
            "lon": ("lon", lon, {"long_name": "longitude", "units": "degree_east"}),
            "time": ("time", time, {"long_name": "Time"}) 
        },
        
        attrs={"model_name": model_name}  # Global attribute for the model name
    )
    
    # Save the dataset to a NetCDF file
    ds.time.encoding["units"] = "days since 1900-01-01"
    ds.to_netcdf(f_out, format="NETCDF4")
    print(f"NetCDF file '{f_out}' has been created successfully.")

lat = ds_tasmax.coords['lat'].values  # Latitude coordinates
lon = ds_tasmax.coords['lon'].values  # Longitude coordinates
time = pd.to_datetime([f'{y}-06-01'])  # Ensure it's a 1D array

# Here, `HW_persistence`, `HW_HWMI`, etc. should be the arrays with the computed metrics
save_hw_metrics_to_netcdf(
    f_out,
    lat,
    lon,
    time,
    hw_metrics[0].values, 
    hw_metrics[1].values, 
    hw_metrics[2].values, 
    hw_metrics[3].values, 
    hw_metrics[4].values, 
    hw_metrics[5].values, 
    hw_metrics[6].values, 
    hw_metrics[7].values,
    model_name=models
)
# - - - - - - - - - - - - - - - - -


# Make some plots 
ds=xr.open_dataset(f'{f_out}')

fig, axs = plt.subplots(ncols=2,nrows=3,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(16, 8))

P1=axs[0,0].pcolormesh(ds.lon,ds.lat,ds.mean_tmax-273.16,transform=ccrs.PlateCarree(),cmap='jet')
cb = fig.colorbar(P1, ax=axs[0,0], orientation='vertical',label='°C',aspect=15,shrink=.75,pad=.015)
axs[0,0].set_title('HW mean tasmax')

P2=axs[0,1].pcolormesh(ds.lon,ds.lat,ds.max_tmax-273.16,transform=ccrs.PlateCarree(),cmap='jet')
cb = fig.colorbar(P2, ax=axs[0,1], orientation='vertical',label='°C',aspect=15,shrink=.75,pad=.015)
axs[0,1].set_title('HW mean tasmax')

P3=axs[1,0].pcolormesh(ds.lon,ds.lat,ds.HWMI,transform=ccrs.PlateCarree(),cmap='jet',vmin=0,vmax=20)
cb = fig.colorbar(P3, ax=axs[1,0], orientation='vertical',label='°C',aspect=15,shrink=.75,pad=.015)
axs[1,0].set_title('HWMId')


P4=axs[1,1].pcolormesh(ds.lon,ds.lat,ds.number,transform=ccrs.PlateCarree(),cmap='jet')
cb = fig.colorbar(P4, ax=axs[1,1], orientation='vertical',label='n.',aspect=15,shrink=.75,pad=.015)
axs[1,1].set_title('HW mean number')

P5=axs[2,0].pcolormesh(ds.lon,ds.lat,ds.hfls_deficit,transform=ccrs.PlateCarree(),cmap='jet',vmin=0,vmax=10)
cb = fig.colorbar(P5, ax=axs[2,0], orientation='vertical',label='',aspect=15,shrink=.75,pad=.015)
axs[2,0].set_title('HW days evaporative deficit')

P6=axs[2,1].pcolormesh(ds.lon,ds.lat,ds.mean_evaporative_fraction,transform=ccrs.PlateCarree(),cmap='jet')
cb = fig.colorbar(P6, ax=axs[2,1], orientation='vertical',label=' ',aspect=15,shrink=.75,pad=.015)
axs[2,1].set_title('HW mean hfls')


for ax in axs.flatten()  : 
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                       color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False      
    
plt.savefig(f'plot_{models}_heatwave_metrics_day_{y}.png')
# - - - - - - - - - - - - - - - - -

