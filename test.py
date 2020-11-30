#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:47:47 2019

@author: qingn
"""

import xarray as xr
import dask
import numpy as np
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
import mpl_toolkits
#import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
import act
import module_ml
from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier
matplotlib.rc('xtick', labelsize=26) 
matplotlib.rc('ytick', labelsize=26) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}
# %%
nav = arm.read_netcdf('/Users/qingn/Desktop/NQ/marnav/*.cdf')
met = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraostry_wd/maraosmetM1.a1.201710*.nc')
#fig=plt.figure(figsize=(22,6))

fig, (ax1, ax2) = plt.subplots(2,1,figsize = (42,6),sharex =True)
#ax = fig.add_subplot(211)
lat = nav['lat'].values
myFmt = DateFormatter("%m/%d-%H") 
time = nav['time']
ax1.xaxis.set_major_formatter(myFmt); 
#    ax.plot_date(time_normal_date,cpc_normal,'.',linewidth=1.5,color='k')
#    ax.plot_date(time_normal_date[S:E], cpc_normal[S:E],'.',linewidth = 0.6, color = 'blue',alpha = 0.1)
#    ax.plot_date(time_normal_date[index_normal],cpc_normal[index_normal],'.',color='r',markersize=4,alpha = 0.1)
ax1.plot_date(time,lat,'.',color='black',markersize=4,alpha = 0.1)

#plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
#ax.set_title('CPC Concentration',fontsize=26)
#ax.set_yscale('log')
ax1.set_ylim([-43.3,-43])
#ax.set_xlim([xmin,xmax])
ax1.yaxis.grid()
ax1.autoscale_view()

ax2 = fig.add_subplot(212)

lon = nav['lon'].values

time = nav['time']
ax2.xaxis.set_major_formatter(myFmt); 
#    ax.plot_date(time_normal_date,cpc_normal,'.',linewidth=1.5,color='k')
#    ax.plot_date(time_normal_date[S:E], cpc_normal[S:E],'.',linewidth = 0.6, color = 'blue',alpha = 0.1)
#    ax.plot_date(time_normal_date[index_normal],cpc_normal[index_normal],'.',color='r',markersize=4,alpha = 0.1)
ax2.plot_date(time,lon,'.',color='blue',markersize=4,alpha = 0.1)

#plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
#ax.set_title('CPC Concentration',fontsize=26)
#ax.set_yscale('log')
#ax.set_ylim([147.25,147.5])
#ax.set_xlim([xmin,xmax])
ax2.yaxis.grid()
ax2.autoscale_view()

# %%
fig=plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax.scatter(lon,lat)
ax.set_ylim([-43.75,-43.5])
ax.set_xlim([147.2,147.4])
