#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 01:24:38 2020
To See the HTDMA figure
@author: qingn
"""

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
def arm_read_netcdf(directory_filebase,time_resolution):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>
    : time_resolution needs to be a string'''
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time=time_resolution).nearest()
    return file
#%% Read 
    
path_htdma = '/Users/qingn/Desktop/NQ/maraoshtdma/maraoshtdmaM1.a1.2017111[2]*.cdf'
htdma= arm.read_netcdf(path_htdma)
#%%
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2017112[6]*.nc'
path_uhsas_clean = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2017111[1]*.nc'
uhsas = arm.read_netcdf(path_uhsas)
uhsas_clean = arm.read_netcdf(path_uhsas_clean)
#%%
fig = plt.figure(figsize =[11,3] )#(FIGWIDTH*1.5,FIGHEIGHT)
ax = plt.gca()
#
pcm = plt.pcolormesh(htdma.aerosol_concentration['time'],htdma.aerosol_concentration.bin,htdma.aerosol_concentration.T,norm=matplotlib.colors.LogNorm(), cmap='jet')
#pcm = ax.pcolormesh(dn_dlnDp_all['time'],lower_bd,dn_dlnDp_all.T,norm=matplotlib.colors.LogNorm(),
#                   cmap='jet')
#plasma,magma [4320:10800]
cbar= fig.colorbar(pcm, ax=ax)
cbar.set_label('Count', rotation=90)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.xaxis_date()
plt.ylabel('Bin number')
fig.autofmt_xdate()
plt.title('AOS HTDMA Size Distribution by bin')
fig.tight_layout() 
plt.show()
#%%
plt.figure(figsize = [11,3])
plt.plot(htdma.dry_diameter_setting.time, htdma.dry_diameter_setting)
plt.xlabel('time')
plt.ylabel('dry_diameter_setting(nm)')
plt.title('DMA1 served as a filter')
##%%
#plt.figure(figsize = [11,3])
#plt.plot(htdma.scan_min_diameter_setting.time, htdma.scan_max_diameter_setting)
#plt.xlabel('time')
#plt.ylabel('dry_diameter_setting(nm)')
##%
plt.figure(figsize = [11,3])
plt.plot(htdma.bin_width.time, htdma.bin_width)
plt.ylabel('bin_width(nm)')
plt.title('60 bins width'' cycling  with time')

plt.figure(figsize = [11,3])
plt.plot(htdma.bin_width.time, htdma.bin_center)
plt.ylabel('bin_center(nm)')
plt.title('60 bins center'' cycling  with time')
#%% UHSAS
#size_dt= uhsas['size_distribution'] # counts
#cpc_1s = arm.read_netcdf(path_cpc)
con = uhsas['concentration'] # Computed concentration 1/cubic centimeter per seconds
con_clean = uhsas_clean['concentration']
lower_bd = uhsas['lower_size_limit'].values[0,]
upper_bd = uhsas['upper_size_limit'].values[0,]
#interval = upper_bd-lower_bd
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
# ACCumulation mode aerosol number
#all_con = np.sum(size_dt,axis =1)
#acc_con = np.sum(size_dt[:,18:],axis=1)
# 11-25 18-11 26 03
#%%

dn_dlogD = con/np.log(interval_meta) #1/cc/nm
dn_dlogD_clean = con_clean/np.log(interval_meta)
#dn_dlnDp_all = con/dlnDp

dn_dlnDp1 = dn_dlogD[4320+300:10800] # 12/19 03 - 12/20 21
dn_dlnDp2 = dn_dlogD[3240:] #12/18 09 - 12/19 24
#dn_dlnDp3 = # 11/27 02-11/28 10
#%%
np.shape(dn_dlnDp1)
plt.figure(figsize=(FIGWIDTH,FIGHEIGHT))
#plt.plot(lower_bd*0.001,dn_dlogD.sum(axis = 0))
plt.plot(lower_bd*0.001,dn_dlogD_clean.mean(axis=0),label = 'non-stack')
#plt.plot(lower_bd*0.001,dn_dlogD.loc['2017-11-26 01':'2017-11-26 03'].mean(axis=0),label = 'stack')
#.loc['2017-11-26 02':'2017-11-26 03']
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylabel('dn/dlnDp(1/cc/nm)')
plt.xlabel('Dp(um)')
plt.title('Measured Size spectra')
#%%
fig.tight_layout() 
plt.show()
fig = plt.figure(figsize = (FIGWIDTH*1.5,FIGHEIGHT))
ax = plt.gca()

pcm = ax.pcolormesh(dn_dlogD['time'].loc['2017-11-26 02':'2017-11-26 03'],lower_bd[:50]/1000,dn_dlogD[:,:50].loc['2017-11-26 02':'2017-11-26 03'].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
#pcm = ax.pcolormesh(dn_dlogD['time'],lower_bd/1000,dn_dlogD.T,norm=matplotlib.colors.LogNorm(),
#                   cmap='jet')
#plasma,magma [4320:10800]
cbar= fig.colorbar(pcm, ax=ax)
cbar.set_label('dn/dlogDp(1/cc/nm)', rotation=90)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
# Setup the DateFormatter for the x axis
#date_format = mdates.DateFormatter('%D')

#ax.xaxis.set_major_formatter(date_format)
#ax.xaxis.set_major_formatter(DateFormatter('%D'))
##column_labels = list('ABCD')
#row_labels = np.arange(uhsas['lower_size_limit'].values[0,0],uhsas['lower_size_limit'].values[0,-1])
#ax.set_yticklabels(row_labels, minor=False)
#ax.set_yticklabels(row_labels, minor=False)
# Rotates the labels to fit

# put the major ticks at the middle of each cell
#ax.set_yticks(np.arange(uhsas['lower_size_limit'].values[0,0],uhsas[/'lower_size_limit'].values[0,-1]) + 0.5, minor=False)

#ax.invert_yaxis()

ax.xaxis_date()
plt.ylabel('Dp(um)')
fig.autofmt_xdate()
plt.title('UHSAS clean')
fig.tight_layout() 
plt.show()
#%%


