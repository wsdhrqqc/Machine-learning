#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:14:52 2019
relashionship between ccn and wind&lat
@author: qingn
"""
import joblib
import seaborn as sns

import xarray as xr
import dask
import numpy as np
import matplotlib.backends.backend_pdf
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
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
#import mpl_toolkits
#import mpl_toolkits.basemap as bm
#from mpl_toolkits.basemap import Basemap, cm
import act
import pathlib, itertools
#import module_ml
#from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot

HOME_DIR = str(pathlib.Path.home()/'Desktop/NQ' )
#%%
def arm_read_netcdf(directory_filebase):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    
    file_dir = str(HOME_DIR + directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time='10s').nearest()
    return file

#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
#%%

ccn_dir = '/marccnspectra/maraosccn1colspectraM1.b1.201712*.nc'
#'/maraoscpc/maraoscpcfM1.b1.%s*'%indate
ccn = arm_read_netcdf(ccn_dir)

exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad01thresh']
time_id = np.array(exhaust_id['time'])
# From the number[0,1,2,3,4...] into the time stamps
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   
# this is aosccn1colspectra-b1-1.2
n_ccn = ccn['N_CCN']
qc_ccn = ccn['qc_N_CCN']
time_ccn = ccn['time'].values
ccn.close()

#%% implement flag on cpc concentration(2)
idx = find_closest(time_id_date, time_ccn) 
flag_ccn = exhaust[idx] #'''This step is slow!!'''
#%%
index_ccn_mad = np.where(flag_ccn == 1)# pick up the contaminated time index
dirty = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn == 0)

clean = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)
n_ccn = np.ma.masked_where(qc_ccn!= 0, n_ccn)
# %% create a dataframe

indate = 201712
date_parser = pd.to_datetime

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

#df = pd.read_csv(infile, parse_dates=['datetime'], date_parser=dateparse)
#'/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.20171030.000030.nc'
wind_lat=pd.read_csv('/Users/qingn/%scpc_wind_lat_.csv'%indate, parse_dates=[0], date_parser=dateparse)

ccn_fullname = '/Users/qingn/%sccn_.csv'%indate
ccn_df = pd.DataFrame({'N_CCN_0.1':n_ccn[:,1][clean],'N_CCN_0.2':n_ccn[:,2][clean] ,'N_CCN_0.5':n_ccn[:,3][clean] ,'N_CCN_0.8':n_ccn[:,4][clean] ,'N_CCN_1.0':n_ccn[:,5][clean]}, index = time_ccn[clean])
#joblib.dump(win_cpc_df1, cpc_wind_fullname)
ccn_df.to_csv(ccn_fullname)
#ccn_201711 = ccn_df
#ccn_df.index = ccn_df.iloc[:,0]
ccn_201711 = pd.read_csv(ccn_fullname, parse_dates=[0], date_parser=dateparse)

columnsNamesArr = wind_lat.columns.values
columnsNamesArr
columnsNamesArr[0]='Time'
wind_lat.set_index('Time')


columnsNamesArr = ccn_201711.columns.values
columnsNamesArr
columnsNamesArr[0]='Time'

ccn_201711.set_index('Time')
trytry = pd.merge_asof(wind_lat, ccn_201711, on = 'Time', tolerance = pd.Timedelta('60m'))
#%%

wind_lat.index = wind_lat.iloc[:,0]

x = wind_lat['cpc_con']
y = wind_lat['wind_speed']
z = wind_lat['wind_direction']

cpc_wind_fullname = '/Users/qingn/%scpc_wind_lat_.csv'%indate
win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean], 'wind_direction':dirt[clean], 'cpc_con':cpc_con1[clean],'lat':nav['lat'][clean]}, index = time_cpc[clean])
#joblib.dump(win_cpc_df1, cpc_wind_fullname)
win_cpc_df1.to_csv(cpc_wind_fullname)

cpc_wind_fullname = '/Users/qingn/%sccn_wind_lat_.csv'%indate
total_df = pd.merge_asof(wind_lat, ccn_df, wind_lat_index = True, ccn_df_index = True)
total_df.to_csv(cpc_wind_fullname)
#%%
x = trytry['N_CCN_1.0']
y = trytry['wind_speed']
z = trytry['wind_direction']
fig, ax = plt.subplots()

im = ax.hexbin(x[x<1000], y[x<1000], gridsize=150, cmap=plt.cm.BuGn)
ax.set_xlim(0,1000)
# bins ='log' 
#plt.xlabel('1/cc')
ax.set_xlabel('1/cc')
#axs[p].set_ylim((0,25))
ax.set_ylabel('m/s')
ax.set_title('wind speed & CPC concentration')
#ax.colorbar()
cb = fig.colorbar(im, ax=ax)
cb.set_label('counts')
#cb.set_label('log10(N)')
plt.show()
#%%
#sns.set()
#g = sns.jointplot(y = 'wind_speed',x='N_CCN_1.0' ,data = trytry[x<1500], color="purple")
#g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)
#
#g = sns.jointplot(x = 'N_CCN_1.0' , y = 'wind_direction', data = trytry,color="lightcoral")
#g.set_axis_labels('cpc concentration(1/cc)','wind direction(degree)', fontweight = 'bold',fontsize = 18)
##
#g = sns.jointplot(x = 'N_CCN_1.0' , y = 'lat', data = trytry,color="blue")
#g.set_axis_labels('cpc concentration(1/cc)','latitude(south_degree)', fontweight = 'bold',fontsize = 18)
##%%
#%%
sns.set()
g = sns.jointplot(x = 'N_CCN_1.0' , y = 'wind_speed', data = trytry[x<1000],bins='log', kind ="hex", color="lightcoral")
#,xlim=[1,1800],ylim=[0.01,25]
g.set_axis_labels('N_CCN_1.0(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)

g = sns.jointplot(x = 'N_CCN_1.0' , y = 'wind_direction', data = trytry[x<1000], bins='log',kind ="hex",color="lightcoral")
g.set_axis_labels('N_CCN_1.0(1/cc)','degree relative to true north', fontweight = 'bold',fontsize = 18)

g = sns.jointplot(x = 'N_CCN_1.0' , y = 'lat', data = trytry[x<1000],bins='log', kind ="hex",color="lightcoral")
g.set_axis_labels('N_CCN_1.0(1/cc)','latitude(south)', fontweight = 'bold',fontsize = 18)
#%%
plt.figure(figsize = [10,5])
plt.plot(ccn['time'],ccn['N_CCN'])
plt.title('N_CCN from maraosccn1colspectraM1')
plt.xlabel('time')
plt.ylabel('1/cc')
#%%
plt.figure(figsize = [10,5])
plt.scatter(ccb_avg['N_CCN'])
plt.title('N_CCN from maraosccn1colavgM1.b1')
plt.xlabel('time')
plt.ylabel('1/cc')