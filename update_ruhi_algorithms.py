#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:54:46 2020
use cpc to change exhaust_flag
@author: qingn
"""

import xarray as xr

import dask
import numpy as np
import numpy.ma as ma
import matplotlib.backends.backend_pdf
import scipy.stats as stats
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
#matplotlib.use('Agg')
import sys
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
#import mpl_toolkits
#import mpl_toolkits.basemap as bm
#from mpl_toolkits.basemap import Basemap, cm
import act
import seaborn as sns
#import module_ml
#from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier

import pathlib
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 18
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}

def arm_read_netcdf(directory_filebase):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time='1h').nearest(tolerance = '2h')
    return file

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
#%%
path_exhaust = '/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc'

cpc = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201711*.nc','1s')
co = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.201711*.nc','10s')



exhaust_id = netCDF4.Dataset(path_exhaust)
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
exhaust_4mad02thresh = exhaust_id['exhaust_4mad02thresh']
#%%
cpc_qc = cpc['qc_concentration']
cpc_con = cpc['concentration']
co_qc=co['qc_co_dry']
co_con = co['co_dry']
#%%
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (12,16))
ax1.plot(cpc_con.time,cpc_con,color= 'red')
ax1.plot(cpc_con[np.where((cpc_qc==0)|(cpc_qc==8)|(cpc_qc==12)|(cpc_qc==192))].time,cpc_con[np.where((cpc_qc==0)|(cpc_qc==8)|(cpc_qc==12)|(cpc_qc==192))] ,'.')

ax2.plot(co_con.time, co_con,color = 'red')
#ax2.plot(co_con[co_con>65536].time, co_con[co_con>65536],'.b')
ax2.plot(co_con[co_qc<16384].time, co_con[co_qc<16384],'.b')
#ax1.set_ylim(bottom=0.0)
ax2.set_ylim(bottom=0.0,top = 1)
ax1.set_ylabel('cpc concentration(#/cc)',fontsize=26,color = 'blue')
ax2.set_ylabel('co mixing ratio(ppmv)',fontsize=26,color = 'blue')
fig.autofmt_xdate()
ax1.set_xlabel('time')
fig.tight_layout()
fig.autofmt_xdate()
ax1.set_title('after_qc_co')
ax2.set_title('after_qc_cpc')
