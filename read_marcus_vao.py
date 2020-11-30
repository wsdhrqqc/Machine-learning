#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:15:41 2020
# Read VAP MARCUS file
@author: qingn
"""


import xarray as xr
from pandas import Grouper
import dask
import numpy as np
from netCDF4 import Dataset
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
import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
import act
import seaborn as sns
#import module_ml
#from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier

import pathlib
FIGWIDTH = 12
FIGHEIGHT = 4 
FONTSIZE = 22
LABELSIZE = 22
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=26) 
matplotlib.rc('ytick', labelsize=26) 
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
#%% Read file and var
vap = Dataset('/Users/qingn/Downloads/Environmental VAP/MARCUS/V1.3/MARCUS_Environmental_Parameters_VAP_V1.3_Voyage1_20171029_20171203.cdf')
height = vap['BT_height']    # 1-7 km, 7 levels
mins_pre = vap['BT_mins_previous'] # three days in total, (minutes previous,time)
lon = vap['BT_longitude']
lat = vap['BT_latitude']
alt = vap['BT_altitude']# (minutes previous, height, time)
# 29 days every 10min, 5 layers and 4176 counts in time
#%% plot BT

#%%
plt.figure(figsize=(FIGWIDTH*1.8,FIGHEIGHT*1.8))
m = Basemap(llcrnrlon=30, llcrnrlat=-80, urcrnrlat=-10, urcrnrlon=180, lat_0=0,
            lon_0=0)
m.drawcoastlines()
m.drawparallels(np.arange(-80.,-20.,20.),labels=[1,1,0,1])
m.drawmeridians(np.arange(30.,181.,20.),labels=[1,1,0,1])
m.shadedrelief()
#i=2
#x, y = m(lonn[date_hobart_davis[2*i]:date_hobart_davis[2*i+1]].values,latt[date_hobart_davis[2*i]:date_hobart_davis[2*i+1]].values)
#m.plot(x,y,label = position[i],color = colors[i])
#%%
m.plot(lon[:48,0,0:880:40],lat[:48,0,100:880:40])