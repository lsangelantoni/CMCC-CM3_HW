#!/usr/bin/env python
import os
os.environ["ESMFMKFILE"] = "/users_home/cmcc/ls21622/.conda/envs/DEVELOP/lib/esmf.mk"
import sys
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
import glob

# Call files with tasmax and hfls statistics for definning HWMId and evaporative deficit. It is already compliant to the considered spatial domain.  
root_clima_stats='/work/cmcc/ls21622/tmp_hw_metrics'
ds_stats=xr.open_dataset(f'{root_clima_stats}/ERA5-land_clima_stats.nc') 
# - - - - - - - - - - - - - - - - -


                         
# From shell
y=                int(sys.argv[1])

lon_min=          int(sys.argv[2])
lon_max=          int(sys.argv[3])
lat_min=          int(sys.argv[4])
lat_max=          int(sys.argv[5])

tasmax_dir=       str(sys.argv[6])
p90_dir=          str(sys.argv[7])
hfls_dir=         str(sys.argv[8])

models=           str(sys.argv[9])
f_out=            str(sys.argv[10])
# - - - - - - - - - - - - - - - - -


# Function to process all the loaded files in
def process_file(file_path, var_name, y, resample=False):
    ds = xr.open_dataset(file_path)
    
    # Rename variables if necessary
    if var_name == 'tasmax':
        ds = ds.rename({'valid_time': 'time', 'longitude': 'lon', 'latitude': 'lat', 't2m': 'tasmax'})
    
    # Sort by latitude
    ds = ds.sortby('lat')

    # Crop domain
    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # Resample to daily resolution (if required)
    if resample:
        ds = ds.resample(time='D').mean('time')
    
    return ds

# Process tasmax files
tasmax_files = []
tasmax_files.extend(glob.glob(f'{tasmax_dir}/*_{y}_0[6-8].nc'))

tasmax_datasets = [process_file(f, 'tasmax', y) for f in tasmax_files]

# Concatenate tasmax datasets along the time dimension
ds_tasmax = xr.concat(tasmax_datasets, dim='time')
ds_tasmax=ds_tasmax.sortby(ds_tasmax.time)
# - - - - - - - - - - - - - - - - -


# Process hfls files
hfls_files = []
hfls_files.extend(glob.glob(f'{hfls_dir}/*_{y}_0[6-8].nc'))
hfls_datasets = [process_file(f, 'hfls', y,resample=True) for f in hfls_files]
# Concatenate hfls datasets along the time dimension
ds_hfls = xr.concat(hfls_datasets, dim='time')
ds_hfls = ds_hfls.sortby(ds_hfls.time)

print('REGRIDDING MOD TO OBS')
method = 'bilinear'
def regrid(ds,ds_out,method):
    regridder = xe.Regridder(ds,ds_out, method = method)
    ds = regridder(ds,ds_out,method)
    return ds

ds_hfls_regrid = regrid(ds_hfls,ds_tasmax,method)
ds_hfls = ds_hfls_regrid
# - - - - - - - - - - - - - - - - -

# Call for tasmax p90:
ds_p90 = xr.open_dataset(f'/data/cmcc/ls21622/ERA5/ERA5_land/tmax/p90/tasmax_ERA5-land_day_1975-1994.nc').rename({'valid_time':'time','longitude':'lon','latitude':'lat'})
ds_p90 = ds_p90.sortby(ds_p90.lat)
ds_p90 = ds_p90.sel(time=ds_p90['time'].dt.month.isin([6, 7, 8]))
ds_p90 = ds_p90.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
# - - - - - - - - - - - - - - - - -



# inputs for HW detection function
ds_tasmax = ds_tasmax.chunk({"time": -1, "lat": 10, "lon": 10}).persist()
ds_hfls = ds_hfls.chunk({"time": -1, "lat": 10, "lon": 10}).persist()
ds_p90 = ds_p90.chunk({"lat": 10, "lon": 10}).persist()


nlat,nlon = np.shape(ds_tasmax.tasmax)[1], np.shape(ds_tasmax.tasmax)[2]
np_tasmax = np.array(ds_tasmax.tasmax)
np_hfls = np.array(ds_hfls.hfls)
np_90p = np.array(ds_p90.tasmax)
np_p25 = np.array(ds_stats.tasmax_clima_p25)
np_iqr_tasmax = np.array(ds_stats.tasmax_clima_iqr)
np_p75 = np.array(ds_stats.hfls_clima_p75)
np_iqr_hfls = np.array(ds_stats.hfls_clima_iqr)
# - - - - - - - - - - - - - - - - -


# Function to compute metrics for a single grid point
@delayed
def compute_metrics(tasmax_series, hfls_series, p90_thresh, p25, iqr_tasmax, p75, iqr_hfls,min_duration=3):
    results = np.full(7, np.nan)  # Store all metrics
    
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
        #print(hw_days)
        hfls_days = np.concatenate([hfls_series[start:end + 1] for start, end in indices])

        results[0] = np.mean(events)                        # Persistence
        results[1] = np.mean(hw_days)                      # Mean Tmax
        results[2] = np.max(hw_days)                       # Max Tmax
        results[3] = len(events)                           # Number of events
        results[4] = np.sum((hw_days - p25) / iqr_tasmax) / len(events)  # HWMI
        results[5] = -1 * np.sum((hfls_days - p75) / iqr_hfls) / len(events)  # HFLS deficit
        results[6] = np.mean(hfls_days)                    # Mean HFLS

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
            p90_thresh = np_90p[:,ii, jj]
            p25 = np_p25[ii, jj]
            iqr_tasmax = np_iqr_tasmax[ii, jj]
            p75 = np_p75[ii, jj]
            iqr_hfls = np_iqr_hfls[ii, jj]

            # Only compute metrics if no NaN is encountered
            tasks.append(compute_metrics(tasmax_series, hfls_series, p90_thresh, p25, iqr_tasmax, p75, iqr_hfls, min_duration=3))

# Filter out None tasks (for NaN grid cells)
tasks = [task for task in tasks if task is not None]

# Compute all tasks
results = da.compute(*tasks)

# Convert results to numpy array and reshape
results = np.array(results)

# Initialize the results array with NaNs
final_results = np.full((7, ds_tasmax.dims['lat'], ds_tasmax.dims['lon']), np.nan)

# Fill in the valid grid points
valid_index = 0
for ii in range(ds_tasmax.dims['lat']):
    for jj in range(ds_tasmax.dims['lon']):
        if not np.isnan(np_tasmax[0, ii, jj]):
            # Place the results into the correct grid point
            final_results[:, ii, jj] = results[valid_index]
            valid_index += 1


hw_metrics = xr.DataArray(
    final_results,
    dims=["metric", "lat", "lon"],
    coords={
        "metric": ["Persistence", "Mean Tmax", "Max Tmax", "Number of events", "HWMI", "HFLS deficit", "Mean HFLS"],
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
    HW_persistence, 
    HW_mean_tmax, 
    HW_max_tmax, 
    HW_number, 
    HW_HWMI, 
    HW_hfls_deficit, 
    HW_mean_hfls,
    model_name
):
    # Create an xarray Dataset with the computed heatwave metrics
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
        },
        coords={
            "lat": ("lat", lat, {"long_name": "latitude", "units": "degree_north"}),
            "lon": ("lon", lon, {"long_name": "longitude", "units": "degree_east"}),
        },
        attrs={"model_name": model_name}  # Global attribute for the model name
    )
    
    # Save the dataset to a NetCDF file
    ds.to_netcdf(f_out, format="NETCDF4")
    print(f"NetCDF file '{f_out}' has been created successfully.")

lat = ds_tasmax.coords['lat'].values  # Latitude coordinates
lon = ds_tasmax.coords['lon'].values  # Longitude coordinates

# Here, `HW_persistence`, `HW_HWMI`, etc. should be the arrays with the computed metrics
save_hw_metrics_to_netcdf(
    f_out,
    lat,
    lon,
    hw_metrics[0].values, 
    hw_metrics[1].values, 
    hw_metrics[2].values, 
    hw_metrics[3].values, 
    hw_metrics[4].values, 
    hw_metrics[5].values, 
    hw_metrics[6].values, 
    model_name="ERA5-land"
)
# - - - - - - - - - - - - - - - - -


# Make some plots
ds=xr.open_dataset(f'{f_out}')

fig, axs = plt.subplots(ncols=1,nrows=1,
                        subplot_kw={'projection': ccrs.PlateCarree()})

P1=axs.pcolormesh(ds_tasmax.lon,ds_tasmax.lat,ds['mean_tmax'],transform=ccrs.PlateCarree())
axs.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
axs.add_feature(cfeature.BORDERS.with_scale('50m'))
#axs.set_extent([-10,30,20,50])
gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                       color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False      

cb = fig.colorbar(P1, ax=(axs), orientation='vertical',label='°C',aspect=15,shrink=.75,pad=.015)
plt.savefig('test_hws.png')
# - - - - - - - - - - - - - - - - -
