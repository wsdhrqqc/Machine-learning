#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:32:16 2019
THis should be done with a few function, I personally begin doing all of this
in a long script and then invide them into differnt part by setting up functions
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
#import mpl_toolkits
#import mpl_toolkits.basemap as bm
#from mpl_toolkits.basemap import Basemap, cm
import act
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

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE) 
# %% Read in data use arm act function
# CPC
cpc = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2018*')
_, index1 = np.unique(cpc['time'], return_index = True)
cpc = cpc.isel(time = index1)
cpc_con = cpc['concentration']
qc_cpc = cpc['qc_concentration']
time_cpc = cpc['time'].values
# EXHAUST_ID
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad02thresh']
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   

# UHSAS
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2018*.nc'
uhsas = arm.read_netcdf(path_uhsas)

_, index1 = np.unique(uhsas['time'], return_index = True)
uhsas = uhsas.isel(time = index1)
uh_con = uhsas['concentration'][:,3:]
time_uhsas = uhsas['time'].values
#%%
'''
We integrate wind and cpc second, to see the wind speed and direction's influence on cpc concentration


met_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.*.nc')
nav_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.*')
nav = nav_ori.resample(time='1s').nearest()
met = met_ori


wspd_name='wind_speed';wdir_name='wind_direction'
heading_name='yaw';cog_name='course_over_ground'
sog_name='speed_over_ground'
# Set variables to be used and convert to radians
rels = met[wspd_name]
unit_ut = met[wspd_name].units
reld = np.deg2rad(met[wdir_name])
reld_deg = met[wdir_name]

head = np.deg2rad(nav[heading_name])
head_deg = nav[heading_name]
cog = np.deg2rad(nav[cog_name])
sog = nav[sog_name]

_, index1 = np.unique(met['time'], return_index = True)
reld_new1 = reld.sel(time=~reld.indexes['time'].duplicated(keep = 'first'))
rels_new1 = rels.sel(time=~reld.indexes['time'].duplicated(keep='first'))
rels_new = rels_new1.resample(time='1s').nearest()
reld_new = reld_new1.resample(time='1s').nearest()
'''

head = np.deg2rad(nav[heading_name])
head_deg = nav[heading_name]
cog = np.deg2rad(nav[cog_name])
sog = nav[sog_name]

#wspd = np.where(sonde['qc_wspd']==0)
#wspd = sonde['wspd'][wspd]
#deg = sonde['deg'][np.where(sonde['qc_deg']==0)]
#%% This is to make sure the time index is monotonic, we only use the first
# duplicated time and the measurements at that time.


'''#rels = rels.resample(time='1min').nearest()'''
#_, index2 = np.unique(reld['time'], return_index = True)
#rels.isel(time = index)

#rels_new = rels.isel(time = index1)
#df3 = reld.loc[~reld.indexes['time'].duplicated(keep = 'first')]
#met = met_ori(time=~met_ori.indexes['time'].duplicated(keep = 'first'))
reld_new1 = reld.sel(time=~reld.indexes['time'].duplicated(keep = 'first'))
#reld_old = reld.sel(time=reld.indexes['time'].duplicated(keep = 'first'))
#reld_new = reld.loc[~reld.indexes['time'].duplicated(keep='first')]
rels_new1 = rels.sel(time=~reld.indexes['time'].duplicated(keep='first'))
#reld_deg_new1 = reld_deg.sel(time=~reld.indexes['time'].duplicated(keep='first'))

rels_new = rels_new1.resample(time='1min').nearest()
reld_new = reld_new1.resample(time='1min').nearest()
#reld_deg_new = reld_deg_new1.resample(time='1min').nearest()

#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
#%% implement flag on cpc concentration(2)
idx2 = find_closest(time_id_date, time_cpc) 

flag_cpc = exhaust[idx2] # to create a relative smaller exhaust_id array
'''This step is slow!!'''

index_cpc_mad = np.where(flag_cpc == 1)# pick up the contaminated time index
dirty2 = np.array(index_cpc_mad[0]) # name the index to be dirty(contaminated index)
index_cpc_clean_mad = np.where(flag_cpc == 0)
clean2 = np.array(index_cpc_clean_mad[0])# name the index to be clean(clean index)
#   %%# implement flag on uhsas total concentration(1)
idx1 = find_closest(time_id_date, time_uhsas)
flag_uhsas = exhaust[idx1]
'''This step is slow!!'''

index_uhsas_mad = np.where(flag_uhsas == 1)# pick up the contaminated time index
dirty1 = np.array(index_uhsas_mad[0])
index_uhsas_clean_mad = np.where(flag_uhsas == 0)
clean1 = np.array(index_uhsas_clean_mad[0])
#%%
lower_bd = uhsas['lower_size_limit'].values[0,]
upper_bd = uhsas['upper_size_limit'].values[0,]
#interval = upper_bd-lower_bd
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
con_non_nan = np.nan_to_num(uh_con) # meta automatically dispeared
total_con = np.dot(con_non_nan, interval_meta[3:]) #unit:1/cc

lower_bd = np.append(lower_bd,[1000.])
dlnDp = np.log(lower_bd[1:]-lower_bd[:-1])
dn_dlnDp_all = uh_con/dlnDp[3:]
#%%
clean_ratio2 = np.size(clean2)/np.size(flag_cpc)
print('cpc clean ratio is',clean_ratio2)
clean_ratio1 = np.size(clean1)/np.size(flag_uhsas)
print('uhsas clean ratio is',clean_ratio1)

qc_zero_ratio = np.size(np.where(qc_cpc==0))/np.size(qc_cpc)
print('uhsas qc==0 ratio is',qc_zero_ratio)
qc_zero_ratio = np.size(np.where(qc_cpc==8))/np.size(qc_cpc)
print('uhsas qc==8 ratio is',qc_zero_ratio)
qc_zero_ratio = np.size(np.where(qc_cpc==962))/np.size(qc_cpc)
print('uhsas qc==962 ratio is',qc_zero_ratio)
#%% Plot figures to compare the contamination-after cpc and uhsas total concentration
S = 86400
E = 86400*1.9
S1 = 8640
E1 = 8640*1.9

myFmt = DateFormatter("%m/%d-%H")
nrows = 2
ncols = 1
N = 86400

fig, axs = plt.subplots(nrows,ncols,figsize = (FIGWIDTH*3.3,FIGHEIGHT*2),sharex =True)
axs = axs.ravel()
fig.subplots_adjust(wspace=.15, hspace=.3)
#fig = plt.figure(figsize= (FIGWIDTH*2,FIGHEIGHT))
p = 0
axs[p].plot_date(time_cpc[clean2[(clean2>S)&(clean2<E)]],cpc_con[clean2[(clean2>S)&(clean2<E)]],
                                      '.',linewidth = 0.6, color = 'yellow', alpha = 0.5,label = 'clean_contaminated')

axs[p].plot_date(time_cpc[dirty2[(dirty2>S)&(dirty2<E)]],cpc_con[dirty2[(dirty2>S)&(dirty2<E)]],
                                      '.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
#fig.tight_layout()
axs[p].set_yscale('log')
fig.tight_layout()
p = 1
#fig = plt.figure(figsize= (FIGWIDTH*2,FIGHEIGHT))
axs[p].plot(time_uhsas[clean1[(clean1>S1)&(clean1<E1)]],cpc_con[clean1[(clean1>S1)&(clean1<E1)]],
                                      '.',linewidth = 0.6, color = 'yellow', alpha = 0.5,label = 'clean_contaminated')

axs[p].plot(time_uhsas[dirty1[(dirty1>S1)&(dirty1<E1)]],cpc_con[dirty1[(dirty1>S1)&(dirty1<E1)]],
                                      '.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
#fig.tight_layout()
axs[p].set_yscale('log')
fig.tight_layout()

#%%
#plt.ioff()
myFmt = DateFormatter("%m/%d-%H")
nrows = 2
ncols = 1
N = 86400
#for i in np.arange(1):
for i in [27,28,29,30]:
    E = int((i+1.0)*1.0*N)
    S = int((i)*1.0*N)
#    S = S_+14400
    E1 = int(E/10.)
    S1 = int(S/10.)
    fig, axs = plt.subplots(nrows,ncols,figsize = (FIGWIDTH*3.3,FIGHEIGHT*2),sharex =True)
    #    fig = plt.figure(figsize = (64,32))
    axs = axs.ravel()
    fig.subplots_adjust(wspace=.15, hspace=.3)
    
    p = 0
    axs[p].plot_date(time_cpc[dirty2[(dirty2>S)&(dirty2<E)]],cpc_con[dirty2[(dirty2>S)&(dirty2<E)]],
                                      '.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_cpc contaminated')
    axs[p].plot_date(time_cpc[clean2[(clean2>S)&(clean2<E)]],cpc_con[clean2[(clean2>S)&(clean2<E)]],
                                      '.',linewidth = 0.6, color = 'black', alpha = 0.5,label = 'mad_cpc clean')
    axs[p].legend()
    axs[p].set_title('CPC concentration')
    axs[p].xaxis.set_major_formatter(myFmt); 
    axs[p].yaxis.grid()
    axs[p].set_ylabel('1/cc')
    #axs[p].set_ylim((0,25))
    axs[p].set_yscale('log')
    
#    a = np.arange(np.where(dirty1==S1)[0][0],np.where(dirty1==E1)[0][0],10)    
    p=1
    axs[p].plot(time_uhsas[dirty1[(dirty1>S1)&(dirty1<E1)]],total_con[dirty1[(dirty1>S1)&(dirty1<E1)]],
                                      '.',linewidth = 0.6, color = 'yellow', alpha = 0.5,label = 'mad_uhsas contaminated')

#    axs[p].plot(time_uhsas[dirty1[a]],total_con[dirty1[a]],
#                                      '.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_cpc contaminated')
    axs[p].plot(time_uhsas[clean1[(clean1>S1)&(clean1<E1)]],total_con[clean1[(clean1>S1)&(clean1<E1)]],
                                      '.',linewidth = 0.6, color = 'green', alpha = 0.5,label = 'mad_uhsas clean')
    axs[p].legend()
    axs[p].set_title('UHSAS total concentration')
    axs[p].xaxis.set_major_formatter(myFmt); 
    axs[p].yaxis.grid()
    axs[p].set_ylabel('1/cc')
    #axs[p].set_ylim((0,25))
    axs[p].set_yscale('log')
#%%
#np.shape(dn_dlnDp2)
plt.figure(figsize=(FIGWIDTH,FIGHEIGHT))
plt.plot(lower_bd[3:-1]*0.001,dn_dlnDp_all.sum(axis = 0))
plt.yscale('log')
plt.ylabel('dn/dlnDp(cm-3*um-1)')
plt.xlabel('Dp(um)')
plt.title('Measured Size spectra')    
    #%%
#fig.tight_layout()


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
