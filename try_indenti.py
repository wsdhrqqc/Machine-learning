#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:01:22 2019

@author: qingn
"""
import xarray as xr
import dask
import numpy as np
import matplotlib.backends.backend_pdf
import scipy.stats
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


#matplotlib.rc('xtick', labelsize=LABELSIZE) 
#matplotlib.rc('ytick', labelsize=LABELSIZE) 
HOME_DIR = str(pathlib.Path.home()/'Desktop/NQ' )

#%%
ccn_colavg1= arm.read_netcdf('/Users/qingn/Desktop/NQ/try_indentification/maraoppsap1flynn1mM1.c1.20171111.000030.nc')
co= arm.read_netcdf('/Users/qingn/Desktop/NQ/try_indentification/maraoscoM1.b1.2017*')
o3 = arm.read_netcdf('/Users/qingn/Desktop/NQ/try_indentification/maraoso3M1.b1.201*')
co_b = co['co_dry']
o3_b =o3['o3'][np.where(o3['qc_o3']==8)[0]]
plt.figure(figsize=[14,5])
plt.plot(o3_b.time,o3_b)
plt.title('Sample period of Ozone_QC==0',fontsize = 18)
plt.ylabel('Ozone concentration(ppb)')
plt.xlabel('time')


plt.figure(figsize=[14,5])
plt.plot(co_b.time,co_b)
plt.title('Sample period of Ozone_QC==0',fontsize = 18)
plt.ylabel('Ozone concentration(ppb)')
plt.xlabel('time')
#o3_con = o3['o3'][np.where(o3['o3_flags']==b'0C100000')[0]]
#o3_con_cal = o3['o3'][np.where(o3['o3_flags']==b'0C310000')[0]]
#
#o3_con_nor = o3_con[np.where(o3_con<27)[0]]
#plt.plot(o3_con_nor.time[np.where(o3_con_nor>10)[0]],o3_con_nor[np.where(o3_con_nor>10)[0]])
#
#
#
#plt.figure(figsize=[14,5])
##plt.plot(cal)
#plt.plot(o3_con_cal.time,o3_con_cal)
#plt.title('Calibration period of Ozone_0C310000',fontsize = 18)
#plt.ylabel('Ozone concentration(ppb)')
#plt.xlabel('time')
#
#plt.plot(o3['o3'].time[cal])
#plt.title('index of time for calibration')
# 

myFmt = DateFormatter("%m/%d-%H")


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize = (14,16),sharex =True)
#    fig = plt.figure(figsize = (64,32))
#    fig1,ax5 = plt.subplot(5,5,sharex = False)
#    ax5 = fig.add_subplot(5,5, sharex = False)
#    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize = (64,32),sharex =True)

ax1.xaxis.set_major_formatter(myFmt); 

ax1.plot_date(o3_b.time,o3_b,label = 'Ozone_full')
#ax1.title('Sample period of Ozone_QC==0',fontsize = 18)
#ax1.ylabel('Ozone concentration(ppb)')
ax1.set_title('Ozone concentration(ppb)',fontsize=26)
ax1.set_ylabel('O3',fontsize=26)


ax2.plot_date(co_b.time,co_b,label = 'CO')
#ax2.title('Caobon',fontsize = 18,label = 'Ozone_full')
ax2.set_ylabel('CO')
ax2.set_title('Carbon monoxide(ppb)',fontsize=26)



ax1.plot_date(ccn_colavg1['co_dry'].time, ccn_colavg1['co_dry'],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
ax2.plot_date(ccn_colavg1['n2o_dry'].time, ccn_colavg1['n2o_dry'])

plt.figure(figsize = [16,8])
#plt.plot_date(o3['o3'].time[np.where(o3['o3_flags']==b'0C100000')[0]], o3['o3'][np.where(o3['o3_flags']==b'0C100000')[0]])
plt.plot_date(o3['o3'].time[:19000], o3['o3'][:19000])
#%%
#ax3.plot_date()
    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax1.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], cpc_con[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax1.plot_date(time_cpc[c[(c > S)&(c<E)]], cpc_con[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax1.legend(loc =2, markerscale=4., fontsize = 'xx-large')
    ax1.set_yscale('log')
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    ax1.set_title('contamination control',fontsize=26)
    ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
