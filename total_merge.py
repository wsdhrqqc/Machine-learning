#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:35:52 2019
This is a script to merge five csv data to show 201711-201803 data
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
from sklearn.ensemble import RandomForestClassifier
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 18
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['axes.labelsize']=FONTSIZE
plt.rcParams['axes.titlesize']=FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE) 

HOME_DIR = str(pathlib.Path.home()/'Desktop/NQ' )
#total = []
indate_index = [201711,201712,201801,201802, 201803]
total = pd.DataFrame()

for indate in indate_index:
    print(indate)
    total = total.append(pd.read_csv('/Users/qingn/%scpc_wind_lat_.csv'%indate, date_parser=False))
    
#    print(np.size(total))
#%%
#total.index = total.iloc[:,0]

#%%
from scipy.stats import gaussian_kde
x = total['cpc_con']
y = total['wind_speed']
z = total['wind_direction']
fig, ax = plt.subplots()

im = ax.hexbin(x, y, gridsize=25, cmap=plt.cm.BuGn)
#ax.set_xlim(0,2000)
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
sns.set()
g = sns.jointplot(y = 'wind_speed',x='cpc_con' ,data = total, color="purple")
g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)

g = sns.jointplot(x = 'cpc_con' , y = 'wind_direction', data = total,color="lightcoral")
g.set_axis_labels('cpc concentration(1/cc)','wind direction(degree)', fontweight = 'bold',fontsize = 18)
#
g = sns.jointplot(x = 'cpc_con' , y = 'lat', data = total,color="blue")
g.set_axis_labels('cpc concentration(1/cc)','latitude(south_degree)', fontweight = 'bold',fontsize = 18)
#%%
sns.set()
g = sns.jointplot(x = 'cpc_con' , y = 'wind_speed', data = total, kind ="hex", color="lightcoral")
#,xlim=[1,1800],ylim=[0.01,25]
g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)
g = sns.jointplot(x = 'cpc_con' , y = 'wind_direction', data = total, kind ="hex",color="lightcoral")
g.set_axis_labels('cpc concentration(1/cc)','degree relative to true north', fontweight = 'bold',fontsize = 18)

g = sns.jointplot(x = 'cpc_con' , y = 'lat', data = total, kind ="hex",color="lightcoral")
g.set_axis_labels('cpc concentration(1/cc)','latitude(south)', fontweight = 'bold',fontsize = 18)
#%%
sns.set()
g=sns.jointplot(x = 'cpc_con', y = 'wind_speed', data = total,kind = "kde", color="purple") # contour plot
g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)
g=sns.jointplot(x = 'cpc_con', y = 'wind_direction', data = total,kind = "kde", color="purple") # contour plot
g.set_axis_labels('cpc concentration(1/cc)','degree relative to true north', fontweight = 'bold',fontsize = 18)

g=sns.jointplot(x = 'cpc_con', y = 'lat', data = total,kind = "kde", color="purple") # contour plot
g.set_axis_labels('cpc concentration(1/cc)','south latitude', fontweight = 'bold',fontsize = 18)

