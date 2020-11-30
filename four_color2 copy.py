#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:26:11 2019

@author: qingn
"""

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
import mpl_toolkits
#import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
import act
import module_ml
from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}

datastream = 'maraoscpc'
var = 'concentration'

# %%
# read in file as objext
cpc = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraos*.nc')
# cpc from 2017-10-29T00:00:00.360000000 to 2018-03-24T23:59:59.430000000
#exhaust_id = arm.read_netcdf('/Users/qingn/Desktop/NQ/exhaust_id/AAS*.nc')
co = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosco/mar*.nc')
# co from 2017-10-29T00:00:00.617000000 to 2018-03-24T19:59:59.200000000
# exhaust from 2017-10-18T23:45:06.000000000 to 2018-03-25T23:59:59.000000000
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
met = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosmetM1.a1/maraosmet*.nc')
nav = arm.read_netcdf('/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.201*')
# %% related to ccn
ccn_colavgfeb = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccnfeb/maraosccn*.nc')
ccn_colavg = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccn/maraosccn*.nc')
ccn100 = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosccn100M1/maraosccn100M1.a1*.nc')
ccncolspectra = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosccn1colspectraM1/maraosccn1colspectraM1*.nc')
ccncol = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosccn1colM1/maraosccn1colM1*.nc')
# %%
# read in time variables
time_cpc = cpc['time'].values
time_id = np.array(exhaust_id['time'])
time_co = co['time'].values
time_met = met['time'].values
#time_ccn_col = ccncol['time'].values


ws = met['wind_speed'].values
wd = met['wind_direction'].values
#n_ccn = ccncol['N_CCN'].values 
# change the time into same system
dt = np.dtype(np.int64)
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
time_id_date = time_id_date.values

# read in variables
exhaust = exhaust_id['exhaust_4mad02thresh'][:]
cpc_con = cpc['concentration']
co_con = co['co_dry']
# %%

fig, (ax1) = plt.subplots(1,1,figsize = (12,6),sharex =True)
ax1.plot(ccn_colavg['droplet_size'],a)

#ax1.set_xlim([273.15,300])
ax1.set_title('CCN size distribution(whole_voyage)',fontsize=20)
ax1.set_ylabel('counts(#after_avg)',fontsize=20)
ax1.set_xlabel('droplet_size(um)',fontsize=20)
#ax1.legend(loc =2, markerscale=4., fontsize = 'xx-large')

#plt.title('CCN size distribution')
#plt.plot(ccn_colavg['droplet_size'],a)



# %%

def intersect_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
#    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]
    return c
''' This is an example showing us how to use the intersect_mtlab
'''
# %%
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


# %%
    
'''
A= np.arange(0, 20.)
target = np.array([[-2, 100., 2., 2.4, 2.5, 2.6]])
print(A)
find_closest(A, target)

OUTPUT : array([[ 0, 19,  2,  2,  3,  3]])
'''

# cc, iaa, ibb = intersect_mtlb(time_id, time_cpc_1)

# find the nearest spot in time_id_date idx1
idx1 = find_closest(time_co, time_cpc)
idx2 = find_closest(time_id_date, time_cpc)
idx3 = find_closest(time_met, time_cpc)
#idx4 = find_closest(time_ccn_col, time_cpc)
ws1 = ws[idx3]
wd1 = wd[idx3]
#n_ccn1 = n_ccn[idx4]
flag_cpc = exhaust[idx2]
co_con1 =co_con[idx1] 
#flag_co = flag_[idx1]

#index_co_mad = np.where(flag_co == 1)
index_cpc_mad = np.where(flag_cpc == 1)
flag_dataarray_cpc = xr.DataArray(flag_cpc, coords=[time_cpc], dims=['time'])
#flag_dataarray_co = xr.DataArray(flag_co, coords=[time_co], dims=['time'])
# %%


index_cpc_ml= module_ml.machine_learning(''.join(['./',datastream,'/*','20171111','*']) ,''.join(['./',datastream,'/mar*.nc'])) 

c = intersect_mtlb(index_cpc_mad, index_cpc_ml)

# %%

index_cpc_ml = np.array(index_cpc_ml[0])
index_cpc_mad = np.array(index_cpc_mad[0])
#index_co_mad = np.array(index_co_mad)

## %%
#myFmt = DateFormatter("%m/%d-%H")
#fig, (ax1, ax2) = plt.subplots(2,1,figsize = (32,16),sharex =True)
##fig = plt.figure(figsize = (32,16))
##ax = fig.add_subplot(211)
##ax1.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
#ax1.xaxis.set_major_formatter(myFmt); 
#ax1.plot_date(time_cpc, cpc_con,'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#
#ax1.plot_date(time_cpc[index_cpc_mad], cpc_con[index_cpc_mad],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
#ax1.plot_date(time_cpc[index_cpc_ml], cpc_con[index_cpc_ml],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
#ax1.plot_date(time_cpc[c], cpc_con[c],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
##[flag[(flag> S)&(flag<E)]]
##ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
##ax.legend(prop={'size': 26})
#ax1.legend(loc =2, markerscale=4., fontsize = 'xx-large')
##np.where(flag<N)
##plt.plot_date(time_normal_date[flag],cpc_normal[flag],'.',color = 'green')
##ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
##plt.xlabel('normal datetime')
#ax1.set_yscale('log')
##ax1.gca().set_ylim(bottom=0.01)
#ax1.yaxis.grid()
#ax1.set_title('contamination control',fontsize=26)
#ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
#ax1.set_ylim([0.1,1000000])
##plt.plot.rcParams.update(params)
#
##ax = fig.add_subplot(212)
#plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
#ax2.xaxis.set_major_formatter(myFmt); 
#ax2.plot_date(time_co, co_con,'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#ax2.plot_date(time_cpc[index_cpc_mad], co_con1[index_cpc_mad],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
#ax2.plot_date(time_cpc[index_cpc_ml], co_con1[index_cpc_ml],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
#ax2.plot_date(time_cpc[c], co_con1[c],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
##[flag[(flag> S)&(flag<E)]]
##ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
##ax.legend(prop={'size': 26})
#ax2.legend(loc =2,  markerscale=4., fontsize = 'xx-large')
#ax2.set_yscale('log')
##np.where(flag<N)
##plt.plot_date(time_normal_date[flag],cpc_normal[flag],'.',color = 'green')
##ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
##plt.xlabel('normal datetime')
#
#plt.gca().set_ylim(bottom=0.01)
#ax2.yaxis.grid()
##    ax.set_ylim([0.1,1000000])
##ax2.set_ylim([0.1,5])
##    ax.set_title('contamination control',fontsize=26)
#ax2.set_ylabel('co (ppmv)',fontsize=26)
#ax2.autoscale_view()
#fig.tight_layout()
##plt.show()
# %%
#plt.ioff()
N = 86400
for i in ([111,112,113,114,115]):
    E = int((i+1.0)*1.0*N)
    S = int((i)*1.0*N)

    myFmt = DateFormatter("%m/%d-%H")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize = (64,64),sharex =True)
#    fig = plt.figure(figsize = (64,32))
#    fig1,ax5 = plt.subplot(5,5,sharex = False)
#    ax5 = fig.add_subplot(5,5, sharex = False)
#    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize = (64,32),sharex =True)
    ax1.xaxis.set_major_formatter(myFmt); 
    ax1.plot_date(time_cpc[S:E], cpc_con[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax1.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], cpc_con[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax1.plot_date(time_cpc[c[(c > S)&(c<E)]], cpc_con[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax1.legend(loc =2, markerscale=4., fontsize = 'xx-large')
    ax1.set_yscale('log')
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    ax1.set_title('contamination control',fontsize=26)
    ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
    ax1.set_ylim([0.1,1000000])

    
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax2.xaxis.set_major_formatter(myFmt); 
    ax2.plot_date(time_cpc[S:E], co_con1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
    ax2.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], co_con1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax2.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], co_con1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax2.plot_date(time_cpc[c[(c > S)&(c<E)]], co_con1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax2.legend(loc =2, markerscale=4., fontsize = 'xx-large')
#    ax2.set_yscale('log')
    plt.gca().set_ylim(bottom=0.01)
    ax2.yaxis.grid()
    ax2.xaxis.grid()
    ax2.set_ylabel('co (ppmv)',fontsize=26)
    ax2.autoscale_view()
#    fig.tight_layout()


    ax3.xaxis.set_major_formatter(myFmt); 
    ax3.plot_date(time_cpc[S:E], ws1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'wind_speed_be_left_data')
    ax3.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], ws1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
#    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax3.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], ws1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax3.plot_date(time_cpc[c[(c > S)&(c<E)]], ws1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax3.legend(loc =2, markerscale=4., fontsize = 'xx-large')
#    ax3.set_yscale('log')
    ax3.xaxis.grid()
#    ax1.set_title('contamination control',fontsize=26)
    ax3.set_ylabel('wind speed (m/s)',fontsize=26)
    ax3.set_ylim([0.1,30])
    
    ax4.xaxis.set_major_formatter(myFmt); 
    ax4.plot_date(time_cpc[S:E], wd1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'wind_direction_be_left_data')
    
    ax4.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], wd1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax4.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], wd1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax4.plot_date(time_cpc[c[(c > S)&(c<E)]], wd1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax4.legend(loc =2, markerscale=4., fontsize = 'xx-large')
#    ax3.set_yscale('log')
    ax4.xaxis.grid()
#    ax1.set_title('contamination control',fontsize=26)
    ax4.set_ylabel('wind direction(degree)',fontsize=26)
    ax4.autoscale_view()
    ax4.set_ylim([-50,360])
    fig.tight_layout()
    # %%
    plt.plot(nav['lon'][432000:864000], nav['lat'][432000:864000])
#    plt.plot(nav['lon'][0:864000], nav['lat'][0:864000])
    plt.plot(nav['lon'][0:432000], nav['lat'][0:432000])
# %%
#     %%
    
pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
out_pdf = r'./ml_mad_aos_ws_wd_nav/test.pdf'
N = 86400
for i in np.arange(4):
    E = int((i+1.0)*N)
    S = int((i)*1.0*N)

    myFmt = DateFormatter("%m/%d-%H")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize = (32,16),sharex =True)
#    fig = plt.figure(figsize = (64,32))
#    fig1,ax5 = plt.subplot(5,5,sharex = False)
#    ax5 = fig.add_subplot(5,5, sharex = False)
#    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize = (64,32),sharex =True)
    ax1.xaxis.set_major_formatter(myFmt); 
    ax1.plot_date(time_cpc[S:E+1], cpc_con[S:E+1],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax1.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], cpc_con[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax1.plot_date(time_cpc[c[(c > S)&(c<E)]], cpc_con[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax1.legend(loc =2, markerscale=4., fontsize = 'x-large')
    ax1.set_yscale('log')
    ax1.autoscale_view()
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    ax1.set_title('contamination control',fontsize=20)
    ax1.set_ylabel('cpc_log (1/cm^3)',fontsize=20)
    ax1.set_ylim([0.1,1000000])

    
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax2.xaxis.set_major_formatter(myFmt); 
    ax2.plot_date(time_cpc[S:E], co_con1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
    ax2.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], co_con1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax2.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], co_con1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax2.plot_date(time_cpc[c[(c > S)&(c<E)]], co_con1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax2.legend(loc =2, markerscale=4., fontsize = 'x-large')
#    ax2.set_yscale('log')
    plt.gca().set_ylim(bottom=0.01)
    ax2.yaxis.grid()
    ax2.xaxis.grid()
    ax2.set_ylabel('co (ppmv)',fontsize=20)
    ax2.autoscale_view()
#    fig.tight_layout()


    ax3.xaxis.set_major_formatter(myFmt); 
    ax3.plot_date(time_cpc[S:E], ws1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'wind_speed_be_left_data')
    ax3.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], ws1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
#    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax3.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], ws1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax3.plot_date(time_cpc[c[(c > S)&(c<E)]], ws1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax3.legend(loc =2, markerscale=4., fontsize = 'x-large')
#    ax3.set_yscale('log')
    ax3.xaxis.grid()
    ax3.yaxis.grid()
    ax3.autoscale_view()
#    ax1.set_title('contamination control',fontsize=26)
    ax3.set_ylabel('wind speed (m/s)',fontsize=26)
    ax3.set_ylim([0.1,30])
    
    ax4.xaxis.set_major_formatter(myFmt); 
    ax4.plot_date(time_cpc[S:E], wd1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'wind_direction_be_left_data')
    
    ax4.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], wd1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax4.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], wd1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax4.plot_date(time_cpc[c[(c > S)&(c<E)]], wd1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax4.legend(loc =2, markerscale=4., fontsize = 'x-large')
#    ax3.set_yscale('log')
    ax4.xaxis.grid()
    ax4.yaxis.grid()
#    ax1.set_title('contamination control',fontsize=26)
    ax4.set_ylabel('wind direction(degree)',fontsize=26)
    ax4.autoscale_view()
    ax4.set_ylim([-50,360])
    fig.tight_layout()
#    
#    f,(ax5,ax6,ax7,ax8) = plt.subplots(1,4,figsize = (64,8), sharex=True, sharey=True)
    f,(ax5,ax6,ax7,ax8) = plt.subplots(1,4,figsize = (32,5),constrained_layout=True)
    f.suptitle('ship_track_6h',fontsize = 20)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
#    plt.xlabel('longitude',fontsize = 20)
#    plt.ylabel('latitude',fontsize=20)
#    ax5.set_title('ship_track',fontsize=20)

 
#    ax5.xaxis.set_major_formatter(myFmt); 
    ax5.plot(nav['lon'][10*S:(10*S+216000):600], nav['lat'][10*S:(10*S+216000):600], label = 'track')
    ax5.set_xlabel('longitude',fontsize = 20)
    ax5.set_ylabel('latitude',fontsize=20)
    ax6.plot(nav['lon'][(10*S+216000):(10*S+216000*2):600], nav['lat'][(10*S+216000):(10*S+216000*2):600], label = 'track')
    ax6.set_xlabel('longitude',fontsize = 20)
    plt.ylabel('latitude',fontsize=20)
    ax7.plot(nav['lon'][(10*S+216000*2):(10*S+216000*3):600], nav['lat'][(10*S+216000*2):(10*S+216000*3):600], label = 'track')
    ax7.set_xlabel('longitude',fontsize = 20)
    plt.ylabel('latitude',fontsize=20)
    
    ax8.plot(nav['lon'][(10*S+216000*3):(10*S+216000*4):600], nav['lat'][(10*S+216000*3):(10*S+216000*4):600], label = 'track')
    ax8.set_xlabel('longitude',fontsize = 20)
    plt.ylabel('latitude',fontsize=20)
    f.autofmt_xdate()
    f.tight_layout()
#    plt.subplot_tool()
#    plt.show()
    #ax5.plot_date([S:E], wd1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'wind_direction_be_left_data')
#    ax5.legend(loc =2, markerscale=4., fontsize = 'xx-large')
#    ax3.set_yscale('log')
#    ax5.xaxis.grid()

#    f.autoscale_view()
    
#    ax4.set_ylim([-50,360])
    cwd = os.getcwd()
    fdir=cwd+'/'+'ml_mad_aos_ws_wd_nav/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'aos_ship_exhaust_ml_wswd_nav'+str(i)+'_'+str(i+1)+'.png')
    plt.show()
    plt.gcf()
#    plt.savefig(fdir+'aos_ship_exhaust_ml_wswd_nav'+str(i)+'_'+str(i+1)+'.png')  
    pdf.savefig(f)
#    pdf.savefig(fig)
pdf.close()
# %%
    fign=plt.figure(figsize=(12,15))
    axn = plt.gca()
#ax = fig.add_subplot(211)
#ax.plot_date(time_id_datetime[begin_second1: begin_second2],best_eval[begin_second1: begin_second2],'.',linewidth=1.5,color='k')
#    ax.plot_date(time[index],data[index],'.',color='r',markersize=4)
#    axn.plot_date(ccn_colavg['time'][np.where(ccn_colavg['N_CCN']<1000)], ccn_colavg['N_CCN'][np.where(ccn_colavg['aerosol_number_concentration']<1000)])    
    axn.plot_date(ccn_colavg['time'], ccn_colavg['N_CCN'],'o','b')
    axn.plot_date(ccn_colavg['time'],ccncol['N_CCN'],'*','r')    
    axn.set_title('CCN Concentration')
#    ax.set_yscale('log')
    axn.set_ylim([0,1000])
#    axn.set_xlim([xmin,xmax])
    axn.xaxis.set_major_formatter(myFmt); 
    fign.autofmt_xdate()
#    fign, axn = plt.subplot(1,1,figsize = (12,6))
    
    
# %% Compare CCN and CPC

for i in np.arange(4):
    E = int((i+1.0)*1.0*N)
    S = int((i)*1.0*N)

    myFmt = DateFormatter("%m/%d-%H")
    fig, (ax1, ax2) = plt.subplots(2,1,figsize = (15,12),sharex =True)
    ax1.xaxis.set_major_formatter(myFmt); 
    ax1.plot_date(time_cpc[S:E], cpc_con[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
    
    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax1.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], cpc_con[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax1.plot_date(time_cpc[c[(c > S)&(c<E)]], cpc_con[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax1.legend(loc =2, markerscale=4., fontsize = 'xx-large')
    ax1.set_yscale('log')
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    ax1.set_title('contamination control',fontsize=26)
    ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
    ax1.set_ylim([0.1,1000000])

    
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax2.xaxis.set_major_formatter(myFmt); 
    ax2.plot_date(time_cpc[S:E], n_ccn1[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
    ax2.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], n_ccn1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
    ax2.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], n_ccn1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_contaminated')
    ax2.plot_date(time_cpc[c[(c > S)&(c < E)]], n_ccn1[c[(c > S)&(c < E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_contaminated')
    ax2.legend(loc =2, markerscale=4., fontsize = 'xx-large')
#   ax2.set_yscale('log')
    plt.gca().set_ylim(bottom=0.01)
    ax2.yaxis.grid()
    ax2.xaxis.grid()
    ax2.set_ylabel('ccn (ppmv)',fontsize=26)
    ax2.autoscale_view()
#    fig.tight_layout()


  
#     %%
    cwd = os.getcwd()
    fdir=cwd+'/'+'ml_mad_aos_ws_wd/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'aos_ship_exhaust_ml_wswd'+str(i)+'_'+str(i+1)+'.png')
    fig.tight_layout()
    


# %%

##    ax = fig.add_subplot(211)
#    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
#    ax1.xaxis.set_major_formatter(myFmt); 
#    plt.plot_date(time_cpc[S:E], cpc_con[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
#    ax1.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], cpc_con[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_bad')
#    ax1.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], cpc_con[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_bad')
#    ax1.plot_date(time_cpc[c[(c > S)&(c<E)]], cpc_con[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_bad')
#    #[flag[(flag> S)&(flag<E)]]
#    #ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
#    #ax.legend(prop={'size': 26})
#    ax1.legend(loc =2, fontsize = 'x-large')
#    #np.where(flag<N)
#    #plt.plot_date(time_normal_date[flag],cpc_normal[flag],'.',color = 'green')
#    #ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
#    #plt.xlabel('normal datetime')
##    ax.set_yscale('log')
#    plt.gca().set_ylim(bottom=0.01)
#    ax1.yaxis.grid()
#    ax1.set_title('contamination control',fontsize=26)
#    ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
#    ax1.autoscale_view()
#    #plt.plot.rcParams.update(params)
#    
##    ax = fig.add_subplot(212)
#    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
#    ax2.xaxis.set_major_formatter(myFmt); 
#    plt.plot_date(time_co[S:E], co_con[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
#    ax2.plot_date(time_cpc[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]], co_con1[index_cpc_mad[(index_cpc_mad>S)&(index_cpc_mad<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_bad')
#    ax2.plot_date(time_cpc[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]], co_con1[index_cpc_ml[(index_cpc_ml>S)&(index_cpc_ml<E)]],'.',linewidth = 0.4, color = 'yellow',alpha = 0.5,label = 'ml_bad')
#    ax2.plot_date(time_cpc[c[(c > S)&(c<E)]], co_con1[c[(c> S)&(c<E)]],'.',linewidth = 0.4, color = 'green',alpha = 0.5, label = 'both_bad')
#    #[flag[(flag> S)&(flag<E)]]
#    #ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
#    #ax.legend(prop={'size': 26})
#    ax2.legend(loc =2, fontsize = 'xx-large')
#    #np.where(flag<N)
#    #plt.plot_date(time_normal_date[flag],cpc_normal[flag],'.',color = 'green')
#    #ax.plot_date(time_normal_date[flag[(flag> S)&(flag<E)]],cpc_normal[flag[(flag> S)&(flag<E)]],'.',color = 'green')
#    #plt.xlabel('normal datetime')
##    ax.set_yscale('log')
#    plt.gca().set_ylim(bottom=0.01)
#    ax2.yaxis.grid()
##    ax.set_ylim([0.1,1000000])
##    ax.set_ylim([0.1,30000])
##    ax.set_title('contamination control',fontsize=26)
#    ax2.set_ylabel('co (ppmv)',fontsize=26)
#    ax2.autoscale_view()
#    

# %%Histagram
''' Group Things 
'''
#np.histogram(ws1[index_cpc_mad], bins = [5,10,15,20,25,30])
fig1 = plt.hist(ws1[index_cpc_mad], bins = [5,10,15,20,25,30], color='#0504aa' ,alpha=0.7, rwidth=0.85)
plt.hist(np.delete(ws1,ws1[index_cpc_mad]),bins = [5,10,15,20,25,30], color='#0504aa' ,alpha=0.7, rwidth=0.85)
         
plt.grid(axis='y', alpha=0.75)
plt.xlabel('wind speed(m/s)',fontsize=26)
plt.ylabel('Frequency',fontsize=26)
#plt.legend()
plt.title('wind speed in free&contamination',fontsize=26)
fig.savefig(fdir+'aos_ship_sta_ws.png')  
             # %%     
                  
plt.hist(wd1[index_cpc_mad], bins = [0,30,60,90,120,150,180,210,240,270,300,330,360], color='#0504aa' ,alpha=0.7, rwidth=0.85)
plt.hist(np.delete(wd1,wd1[index_cpc_mad]), bins = [0,30,60,90,120,150,180,210,240,270,300,330,360], color='#0504aa' ,alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('wind direction(degree)',fontsize=26)
plt.ylabel('Frequency',fontsize=26)
#plt.legend()
plt.title('wind speed in free&contamination',fontsize=26)
fig.savefig(fdir+'aos_ship_sta_wd.png')  
#wd1
#    
    
# %% statistical analysis
    
both_bad_ratio = np.size(c)/np.size(time_cpc)
print('both_bad_ratio: '+ str(both_bad_ratio))
mad_bad_ratio = np.size(index_cpc_mad)/np.size(time_cpc)
print('Ruhi_MAD_bad_ratio: '+ str(mad_bad_ratio))
ml_bad_ratio = np.size(index_cpc_ml)/np.size(time_cpc)
print('Machine_Learning_bad_ratio: '+ str(ml_bad_ratio))