#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 07:26:47 2019
Check UHSAS (Ultra-High Sensitivity Aerosol) Bin=99
@author: qingn
"""
import xarray as xr
import pyproj
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
import glob
#from shapely.geometry import Point
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
import mpl_toolkits
import act.io.armfiles as arm
import act.plotting.plot as armplot
import os, re, fnmatch
#from mpl_toolkits.basemap import Basemap
import pathlib, itertools, time
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 18
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE) 
#%%
dir_name = str(pathlib.Path.home()/'Desktop/NQ/maraosuhsas')

exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201*'
#path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2017112[5-6]*'
#/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.20171029.000033.nc
path_ccn_spectra = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.2017112[5-6]*'
#path_ccn_avg = '/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1.2017121[8-9]*'
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2017112[5-6]*.nc'
path_co = '/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.2017112[5-6]*'
#file_sample = glob.glob(dir_name+'/'+'maraosuhsasM1.a1.2017111[2-4]*')
#path_sonde = '/Users/qingn/Desktop/NQ/sounde/marsondewnpnM1.b1.201*'
#ccn_avg = arm.read_netcdf(path_ccn_avg)
uhsas = arm.read_netcdf(path_uhsas)
#ccn = arm.read_netcdf(path_ccn_spectra)
size_dt= uhsas['size_distribution'] # counts
#cpc_1s = arm.read_netcdf(path_cpc)
con = uhsas['concentration'] # Computed concentration 1/cubic centimeter per seconds
#cpc = arm.read_netcdf(path_cpc).resample(time='10s').nearest()
cpc = arm.read_netcdf(path_cpc)
#cpc = cpc.resample(time='1s').nearest()
#con = con.resample(time = '10s').nearest()
cpc_con = cpc['concentration']
qc_cpc = cpc['qc_concentration']
#co = arm.read_netcdf(path_co)
exhaust = exhaust_id['exhaust_4mad02thresh']
#%%
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   
time_cpc = cpc['time'].values
#%%
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

idx2 = find_closest(time_id_date, time_cpc)
#%%
lower_bd = uhsas['lower_size_limit'].values[0,]
upper_bd = uhsas['upper_size_limit'].values[0,]
#interval = upper_bd-lower_bd
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]

con_non_nan = np.nan_to_num(con) # meta automatically dispeared
total_con = np.dot(con_non_nan, interval_meta) #unit:1/cc
#lower_bd = list(lower_bd)
lower_bd = np.append(lower_bd,[1000.])
dlnDp = np.log(lower_bd[1:]-lower_bd[:-1])

#%%
'''Figure to check consistence between UHSAS and CPC concentration, the UHSAS
total concentration is calculated by integrating all the bins'''
myFmt = DateFormatter("%m/%d-%H")
nrows = 3
ncols = 1


fig, axs = plt.subplots(nrows,ncols,figsize = (FIGWIDTH*3.3,FIGHEIGHT*2),sharex =True)
#    fig = plt.figure(figsize = (64,32))
axs = axs.ravel()
fig.subplots_adjust(wspace=.15, hspace=.3)

p = 0
axs[p].plot(con['time'], total_con, color='black', label='UHSAS 60-1000nm',alpha = 0.5)
#axs[p].plot_date(cpc['time'][np.where((qc_cpc==0)|(qc_cpc==8))], cpc_con[np.where((qc_cpc==0)|(qc_cpc==8))], color = 'blue', label='10-3000nm')
axs[p].plot(cpc['time'][np.where(qc_cpc==0)], cpc_con[np.where(qc_cpc==0)], color = 'blue', label='CPC 10-3000nm',alpha = 0.5)
#axs[p].plot(wspd['time'], wspd,'green',label = 'sonde wind speed')
axs[p].legend()
axs[p].set_title('UHSAS total concentration')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_ylabel('1/cc')
#axs[p].set_ylim((0,25))
axs[p].set_yscale('log')
p = 1
axs[p].plot(cpc_con['time'][np.where((qc_cpc==0)|(qc_cpc==8))],cpc_con[np.where((qc_cpc==0)|(qc_cpc==8))]-total_con[np.where((qc_cpc==0)|(qc_cpc==8))],label = 'diff=cpc-uhsas')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_title('diff_concentration')
axs[p].set_ylabel('1/cc')
axs[p].legend()
axs[p].set_ylim((-3000,3000))
#axs[p].set_yscale('log')
p = 2
axs[p].plot(cpc_con['time'][np.where((qc_cpc==0)|(qc_cpc==8))],cpc_con[np.where((qc_cpc==0)|(qc_cpc==8))]/total_con[np.where((qc_cpc==0)|(qc_cpc==8))],label = 'ratio=cpc/uhsas')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_title('ratio_concentration')
axs[p].set_ylabel('1/cc')
axs[p].legend()
#axs[p].set_yscale('log')
#%%
myFmt = DateFormatter("%m/%d-%H")
nrows = 4
ncols = 1

fig, axs = plt.subplots(nrows,ncols,figsize = (FIGWIDTH*3.3,FIGHEIGHT*2),sharex =True)
#    fig = plt.figure(figsize = (64,32))
axs = axs.ravel()
fig.subplots_adjust(wspace=.15, hspace=.3)

p = 0
axs[p].plot_date(con['time'], total_con, color='black', label='60-1000nm')

#axs[p].plot(wspd['time'], wspd,'green',label = 'sonde wind speed')
axs[p].legend()
axs[p].set_title('UHSAS total concentration')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_ylabel('1/cc')
#axs[p].set_ylim((0,25))
axs[p].set_yscale('log')
p = 1

#axs[p].plot_date(reld['time'][:5000], reld_deg[:5000], color = 'grey', label='wind_direction')
#axs[p].plot_date(dirt['time'][:5000], dirt[:5000],color ='blue',label='true_wind_direction')

axs[p].plot_date(cpc['time'][np.where((qc_cpc==0)|(qc_cpc==8))], cpc_con[np.where((qc_cpc==0)|(qc_cpc==8))], color = 'blue', label='10-3000nm')

#axs[p].plot_date(dirt['time'], dirt ,'blue',label='true_wind_direction')
#axs[p].plot(head['time'], head_deg, 'yellow', label='heading direction')
axs[p].legend()
axs[p].set_yscale('log')
#axs[p].plot_date(deg['time'], deg,color='red')
#axs[p].plot(deg,'green',label = 'sonde wind speed')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_title('CPC concentration')
axs[p].set_ylabel('1/cc')

#axs[p].set_ylim((0,360))
p =2

axs[p].plot_date(cpc_con['time'][np.where((qc_cpc==0)|(qc_cpc==8))],cpc_con[np.where((qc_cpc==0)|(qc_cpc==8))]-total_con[np.where((qc_cpc==0)|(qc_cpc==8))],label = 'cpc-uhsas')
axs[p].xaxis.set_major_formatter(myFmt); 
axs[p].yaxis.grid()
axs[p].set_title('diff_concentration')
axs[p].set_ylabel('1/cc')
axs[p].legend()
#axs[p].set_ylim((-7000,10000))

p=3

axs[p].plot_date(co['time'][np.where(co['qc_co']==0)], co['co'][np.where(co['qc_co']==0)], color = 'green', label='co concentration')
axs[p].yaxis.grid()
axs[p].set_title('co concentration')
axs[p].set_ylabel('ppmv')
axs[p].legend()
#axs[p].set_ylim((0.045,0.065))
fig.tight_layout() 


#%%
#= plt.pcolormesh(size_dt.T)
dn_dlnDp_all = con/dlnDp

dn_dlnDp1 = dn_dlnDp_all[4320+300:10800] # 12/19 03 - 12/20 21
dn_dlnDp2 = dn_dlnDp_all[3240:] #12/18 09 - 12/19 24
#dn_dlnDp3 = # 11/27 02-11/28 10

#%%
fig = plt.figure(figsize = (FIGWIDTH*1.5,FIGHEIGHT))
ax = plt.gca()

pcm = ax.pcolormesh(dn_dlnDp_all['time'],lower_bd,dn_dlnDp_all.T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
#plasma,magma [4320:10800]
cbar= fig.colorbar(pcm, ax=ax)
cbar.set_label('dN/dlogDp', rotation=90)
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
plt.ylabel('Dp(nm)')
fig.autofmt_xdate()
plt.title('UHSAS before applying contamination flag')
fig.tight_layout() 
plt.show()
#%%
np.shape(dn_dlnDp2)
plt.figure(figsize=(FIGWIDTH,FIGHEIGHT))
plt.plot(lower_bd[:-1]*0.001,dn_dlnDp_all.sum(axis = 0))
plt.yscale('log')
plt.ylabel('dn/dlnDp')
plt.xlabel('Dp(um)')
plt.title('Measured Size spectra')
#plt.xscale('log')
#%%
ccn_avg.close()
uhsas.close()
ccn.close()
cpc.close()
co.close()
