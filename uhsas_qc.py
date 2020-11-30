#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:59:24 2020
Check the uhsas and precipitation
UHSAS qc
@author: qingn
"""
import xarray as xr
from pandas import Grouper
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

#matplotlib.use('Agg')
import sys
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
#import mpl_toolkits
#import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
import act
import seaborn as sns
#import module_ml
#from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier
#%%
import pathlib
FIGWIDTH = 12
FIGHEIGHT = 4 
FONTSIZE = 12
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=11) 
matplotlib.rc('ytick', labelsize=11) 
params = {'legend.fontsize': 20,
          'legend.handlelength': 5}
colors = ['cyan','black','yellow','red']

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
home = os.path.expanduser("/Users/qingn/Desktop/NQ")
thesis_path = os.path.join(home,'personal','thesis')
# piclke file data (under NQ)
#pkl = os.path.join(os.getcwd(), "pkl")
#pkl = os.path.join(home, "pkl")
#/Users/qingn/Desktop/NQ/personal/thesis/IMG_5047.jpg

# figure save directory
figpath = os.path.join(thesis_path, "Figures")
#%% Whole dataset accumulation for some reason does not have 
path_uhsas = '/Users/qingn/Desktop/NQ/sdp_aero_136col.csv'
df = pd.read_csv(path_uhsas,parse_dates=True)
#%%
#path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2017112[6]*.nc'
path_uhsas_clean = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2018021[3-9]*.nc'
#uhsas = arm.read_netcdf(path_uhsas)
uhsas_clean = arm_read_netcdf_for_time_resolution(path_uhsas_clean,'10min')

#%% We need a cpc_con
path_='/Users/qingn/Desktop/NQ/ten_10min_after_missingcpc_0_nan.csv'
df_ = pd.read_csv(path_,parse_dates = True)
df_ = df_.set_index('Unnamed: 0')
df_.index = pd.to_datetime(df_.index)
inter = df_['cpc_con'].resample('1min').mean()
cpc = inter.resample('10min').nearest()
del inter
#%%
path__ = '/Users/qingn/Desktop/NQ/aero_modes_rain_wind.csv'
df__ = pd.read_csv(path__,parse_dates = False)
df__.index = pd.to_datetime(df__['Date'])
#%%
#path_='/Users/qingn/Desktop/NQ/ten_10min_after_missingcpc_0_nan.csv'
#df_ = pd.read_csv(path_,parse_dates = True)
uhsas_99 = pd.read_csv('uhsas_99_con.csv',parse_dates = True)
#uhsas_99 = uhsas_99.set_index('time')
uhsas_99['time'] = pd.to_datetime(uhsas_99['time'])
uhsas_99['cpc_con'] = cpc
uhsas = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.20171111*.nc','10s')
bins = uhsas['lower_size_limit'].values[0]

#lower_str = [str(i) for i in bins]
#df__ = df__['2017-10-29 04:00:00':'2018-03-24 14:50:00']
df__['time'] = df__.index
df_sdp_aero= pd.merge_asof(df__,uhsas_99,on = 'time',tolerance = pd.Timedelta('10min'))

key_words = df_sdp_aero.keys()[:]
x_100_350= df_sdp_aero[df_sdp_aero.columns[38:56]].sum(axis=1,min_count=1)
df_sdp_aero['100-350'] = x_100_350

x_350_700= df_sdp_aero[df_sdp_aero.columns[56:101]].sum(axis=1,min_count=1)
df_sdp_aero['350-700'] = x_350_700

x_700_1000= df_sdp_aero[df_sdp_aero.columns[101:137]].sum(axis=1,min_count=1)
df_sdp_aero['700-1000'] = x_700_1000

df_sdp_aero.to_csv('df_sdp_aero.csv',index=True)
#%%
v = np.linspace(-.1, 15.0, 15, endpoint=True)
#plt.contourf(uhsas_clean['concentration'].T,v)
plt.contourf(uhsas_clean['concentration'].time.values,bins,uhsas_clean['concentration'].T,v)
plt.colorbar()









