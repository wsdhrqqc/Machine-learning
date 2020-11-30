#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:14:16 2020

Backup on Github Qing

Calculate speed over ground over the four voyage

@author: qingn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:29:53 2019

@author: qingn
"""
import xarray as xr
import pyproj
#import metpy
import dask
import numpy as np
import act
#import matplotlib as mpl
#import astral
import pandas as pd
from matplotlib.dates import DateFormatter,date2num
import glob
import netCDF4
import os
from act.io.armfiles import read_netcdf
import datetime
import matplotlib
import sys
#from shapely.geometry import Point
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
import mpl_toolkits
import act.io.armfiles as arm
import act.plotting.plot as armplot
#from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import h5py as h5

FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 22
LABELSIZE = 22
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams["legend.framealpha"] = 0.3

matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE) 

def arm_read_netcdf_for_time_resolution(directory_filebase, time_resolution):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time=time_resolution).mean()
    file = file_ori.resample(time=time_resolution).nearest(tolerance = time_resolution)
    return file
#%% Starting from 1029-1203

nav_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.201711*','10s')
nav = nav_ori.resample(time='1min').mean()

wspd_name='wind_speed';wdir_name='wind_direction'
heading_name='yaw';cog_name='course_over_ground'
sog_name='speed_over_ground'
#%%
chl_path = '/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/requested_files/A2017306.L3b_DAY_CHL.x.nc'
chl_l2 = '/Users/qingn/Desktop/NQ/personal/thesis/visst_chl/A2017335033000.L2_LAC_OC.nc'
#'/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/requested_files/A2017305.L3b_DAY_CHL.x.nc'
chl_path ='/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/A2017305.L3m_DAY_CHL_chlor_a_4km.nc'
chl_path = '/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/requested_files 2/A2017302.L3m_DAY_CHL.x_chlor_a.nc'
#%%
chl = netCDF4.Dataset(chl_path)
chl_a = chl['chlor_a']
lon = chl['lon'][:] # 6000: +70
lat = chl['lat'][:] # 3745: -66
#%
plt.contourf(lon,lat,np.log(chl_a[:]))
plt.colorbar()
#%%
chl_viirs = '/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/V2017305.L3m_DAY_SNPP_CHL_chl_ocx_4km.nc'

#chl_path = '/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/A2017305.L3m_DAY_CHL_chlor_a_4km.nc'

chl = netCDF4.Dataset(chl_viirs)
chl_a = chl['chl_ocx']
lon = chl['lon'][:]
lat = chl['lat'][:]
plt.contourf(lon,lat,np.log(chl_a[:]))
plt.colorbar()
#%%
csv = pd.read_csv('/Users/qingn/Desktop/NQ/personal/thesis/chlorophyll1991.csv', parse_dates=True)

lonn = csv['LONG (DEC)']
latt = csv['LAT (DEC)']
z = csv['Chl a']
plt.scatter(lonn,latt,z)
plt.colorbar()
#%%
whole = pd.read_csv('/Users/qingn/Desktop/NQ/aero_stack_uh_wind_rain_loc_wibs_10m.csv',parse_dates = True)
whole = whole.set_index('time')
whole.index =pd.to_datetime(whole.index)
#%%
plt.plot(whole['lat'][whole['int']>1][whole['AA']==0],whole['int'][whole['int']>1][whole['AA']==0],'r*')
plt.plot(whole['lon'][whole['int']>1][whole['AA']==0],whole['int'][whole['int']>1][whole['AA']==0],'b*')

plt.plot(whole['lat'][whole['iten']>1][whole['AA']==0],whole['int'][whole['iten']>1][whole['AA']==0],'r*')
plt.plot(whole['lon'][whole['iten']>1][whole['AA']==0],whole['int'][whole['iten']>1][whole['AA']==0],'b*')

plt.plot(whole['lon'][whole['new_flag']==1],whole['lat'][whole['new_flag']==1],'r.')
plt.plot(whole['lon'][whole['new_flag']==0],whole['lat'][whole['new_flag']==0],'b.')
#%%
path_sonde = '/Users/qingn/Desktop/NQ/sounde/marsondewnpnM1.b1.2018022[1-9]*'
sonde = arm.read_netcdf(path_sonde)
idx_surface = np.where((sonde['alt']>30) and (sonde['alt']<60))
wspd = sonde['wspd'][idx_surface]
#%%
fig = plt.figure()
plt.plot(whole['aos_tr_wind']['2018-02-21':'2018-02-28'].index,whole['aos_tr_wind']['2018-02-21':'2018-02-28'].values,label = 'aos')
plt.plot(whole['aad_tr_wind']['2018-02-21':'2018-02-28'].index,whole['aad_tr_wind']['2018-02-21':'2018-02-28'].values,label = 'aad')
#fig = plt.figure()
plt.plot(wspd['time'].values, wspd.values,'g*',label = 'sonde wind speed')
plt.legend()
fig.autofmt_xdate()
#%%
f = h5.File(chl_path, "r")
    # Get and print list of datasets within the H5 file
datasetNames = [n for n in f.keys()]
for n in datasetNames:
    print(n)
#%%
#    import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

print(find_nearest(chl['lat'], -42))

#%%
location = whole[['lat','lon','AA']][whole['AA']!=1].resample('D').mean()# 123 rows
#%%#% Aqua has 16 days okay

files = glob.glob('/Users/qingn/Desktop/NQ/personal/thesis/Figures/wacr_doppler/requested_files 2/A201*')
files = sorted(files)
len(files)
#% Aqua has 16 days okay
for i in np.arange(0,147):
    chl_path = files[i]
    chl = netCDF4.Dataset(chl_path)
    chl_a = chl['chlor_a']
    lon = chl['lon'][:] # 6000: +70
    lat = chl['lat'][:] # 3745: -66
#    plt.figure()
#    plt.contourf(lon,lat,np.log(chl_a[:]))
#    plt.colorbar()
#    print(location['lat'][i])
    x = find_nearest(lat, location['lat'][i])
    y = find_nearest(lon, location['lon'][i])
    print(i,chl_a[x[0],y[0]],chl_a[x[0],y[0]+1],chl_a[x[0],y[0]-1])
    print(i,chl_a[x[0],y[0]],chl_a[x[0]+1,y[0]],chl_a[x[0]-1,y[0]])
#    print(i,chl_a[x[0],y[0]])

#%%
    # Terra has 17 days
files = glob.glob('/Users/qingn/Desktop/NQ/personal/thesis/terra_chl/requested_files/T201*')
files = sorted(files)
len(files)
#%%
chl_terra = []
time_terra = []
for i in np.arange(0,147):
#    i = 42
    chl_path = files[i]
    chl = netCDF4.Dataset(chl_path)
    chl_a = chl['chlor_a']
    lon = chl['lon'][:] # 6000: +70
    lat = chl['lat'][:] # 3745: -66
#    plt.figure()
#    plt.contourf(lon,lat,np.log(chl_a[:]))
#    plt.colorbar()
#    print(location['lat'][i])
    x = find_nearest(lat, location['lat'][i])
    y = find_nearest(lon, location['lon'][i])
    if  not ma.is_masked(chl_a[x[0],y[0]]):
        print('Yes',i)
        time_terra.append(location.index[i])
        chl_terra.append(chl_a[x[0],y[0]].data.tolist())
    elif not (ma.is_masked(chl_a[x[0],y[0]+1]) and ma.is_masked(chl_a[x[0],y[0]-1]) and ma.is_masked(chl_a[x[0]+1,y[0]]) and ma.is_masked(chl_a[x[0]-1,y[0]])):
        print('Neighbor!',i)
        time_terra.append(location.index[i])
        neighbor = [chl_a[x[0],y[0]+1].data,chl_a[x[0],y[0]-1].data,chl_a[x[0]+1,y[0]].data,chl_a[x[0]-1,y[0]].data]
        chl_terra.append(np.mean(neighbor))
        
#        if
#    print(i,chl_a[x[0],y[0]],chl_a[x[0],y[0]+1],chl_a[x[0],y[0]-1])
#    print(i,chl_a[x[0],y[0]],chl_a[x[0]+1,y[0]],chl_a[x[0]-1,y[0]])
#    print(i,chl_a[x[0],y[0]])
#%% # VISST 13 days
files = glob.glob('/Users/qingn/Desktop/NQ/personal/thesis/visst_chl/requested_files/V201*')
files = sorted(files)
len(files)

for i in np.arange(0,147):
    chl_path = files[i]
    chl = netCDF4.Dataset(chl_path)
    chl_a = chl['chlor_a']
    lon = chl['lon'][:] # 6000: +70
    lat = chl['lat'][:] # 3745: -66
#    plt.figure()
#    plt.contourf(lon,lat,np.log(chl_a[:]))
#    plt.colorbar()
#    print(location['lat'][i])
    x = find_nearest(lat, location['lat'][i])
    y = find_nearest(lon, location['lon'][i])
    print(i,chl_a[x[0],y[0]],chl_a[x[0],y[0]+1],chl_a[x[0],y[0]-1])
    print(i,chl_a[x[0],y[0]],chl_a[x[0]+1,y[0]],chl_a[x[0]-1,y[0]])
    

#%%
plt.figure()
plt.contourf(lon,lat,np.log(chl_a[:]))
plt.colorbar()   
    

    #%%
x = find_nearest(chl['lat'], location['lat'][0])
y = find_nearest(chl['lon'], location['lon'][0])
print(chl_a[x[0],y[0]])
#%%
