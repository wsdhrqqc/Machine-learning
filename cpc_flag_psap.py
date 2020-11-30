#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:14:50 2020

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
matplotlib.use('Agg')
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

#%%
path_impactor = '/Users/qingn/Desktop/NQ/maraosimpactor/maraosimpactorM1.b1.20180323.000000.nc'

path_psap = '/Users/qingn/Desktop/NQ/maraospsap/maraoppsap1flynn1mM1.c1.2017**.nc'
path_exhaust = '/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc'
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201*'
#path_ccnspectra = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.2017111*.nc'
#arm_ds = xr.open_mfdataset(path_exhaust)
#time_id = arm_ds.time
exhaust_id = netCDF4.Dataset(path_exhaust)
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
exhaust_4mad02thresh = exhaust_id['exhaust_4mad02thresh']

slf_time = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-23T23:59:59', freq='s')
#slf_time1 = pd.date_range(start ='2017-10-18T23:45:06',end = '2018-03-25T23:59:59', freq='s')
#flag = [0]*np.size(slf_time)
#mask = slf_time.isin(time_id_date)

exhaust_ds = xr.Dataset({'flag':('time',exhaust_4mad02thresh),'time':time_id_date})
exhaust_ds_full = exhaust_ds.resample(time='1s').nearest(tolerance='10s')
exhaust_ds_full_fill = exhaust_ds_full.fillna(1)
#time_id_values = time_id.values

#%%
impactor = arm.read_netcdf(path_impactor)

psap = arm_read_netcdf(path_psap,'1min')
cpc = arm_read_netcdf(path_cpc,'1s')
cpc_con = ma.masked_where(cpc['qc_concentration']>0,cpc['concentration'])
Ba_time = psap.Ba_B_combined.time
#%%
Ba_B_ori = psap.Ba_B_combined
Ba_G_ori = psap.Ba_G_combined
Ba_R_ori = psap.Ba_R_combined
Ba_B = ma.masked_where(psap.qc_Ba_B_combined>0,psap.Ba_B_combined)
Ba_R = ma.masked_where(psap.qc_Ba_R_combined>0,psap.Ba_R_combined)
Ba_G = ma.masked_where(psap.qc_Ba_G_combined>0,psap.Ba_G_combined)

#%% create a huge dataframe start for the whole marcus journey

df = pd.DataFrame()
# the first number comes from the slf_time[0]  np.where(time_id_date==slf_time[-1])

#df['flag'] = exhaust_ds_full_fill.flag[855937:12843088]
#df['time'] = 

#df['ab_coeff_B'] = Ba_B
df['cpc_con'] = cpc_con 
df = df.set_index(slf_time)
# We slice the exhaust to only include MARCUS authorized period.
A = np.where(exhaust_ds_full_fill.time ==np.datetime64(slf_time[0]))[0][0]
B = np.where(exhaust_ds_full_fill.time ==np.datetime64(slf_time[-1]))[0][0]
df['flag'] = exhaust_ds_full_fill.flag[A:(B+1)]
df['index'] = range(np.size(cpc_con))

# We slice for the psap since it has missing period after 20171214
C = df.loc['2017-12-14T23:59:00.000000000']['index']
C = int(C)
df_psap_1s = df[0:(C+1)]
df_psap = df_psap_1s.resample('1min').nearest()
#%%
df_psap['blue_absb_coeff'] = Ba_B_ori
df_psap['gre_absb_coeff'] = Ba_G_ori
df_psap['red_absb_coeff'] = Ba_R_ori
#%% Figures
# Absorption Coefficient, normally alpha, discribes the intensity attenuation of the light passing through a material.
# The higher alpha, the shorter length the light can penetrate into a material before it is absorbed
myFmt = DateFormatter("%d-%H-%m")
nrows =2
ncols =1

fig, axs = plt.subplots(nrows,ncols,figsize = (FIGWIDTH*3.3,FIGHEIGHT*2),sharex =True)
#    fig = plt.figure(figsize = (64,32))
axs = axs.ravel()
fig.subplots_adjust(wspace=.15, hspace=.3)

date_list=['2017-11-18 09','2017-11-20 00','2017-11-25 22','2017-11-26 02','2017-11-26 03']
j=3
k=4
p = 0
axs[p].plot(df_psap['cpc_con'][date_list[j]:date_list[k]],label = 'cpc')
axs[p].plot(df_psap['cpc_con'][date_list[j]:date_list[k]].where(df_psap['flag']==1), label = 'stack',color = 'red')
axs[p].legend()
axs[p].set_title('Condensation particle counter')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_ylabel('1/cc')
#axs[p].set_ylim((0,25))
axs[p].set_yscale('log')

p=1
axs[p].plot(df_psap['blue_absb_coeff'][date_list[j]:date_list[k]],label = 'blue_464nm',color = 'blue')
axs[p].plot(df_psap['gre_absb_coeff'][date_list[j]:date_list[k]],label = 'green_529nm',color = 'green')
axs[p].plot(df_psap['red_absb_coeff'][date_list[j]:date_list[k]],label='red_648nm',color = 'red')

axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].legend()
axs[p].set_title('PSAP Absorption Coefficient')
axs[p].yaxis.grid()
axs[p].set_ylabel('1/mm')














