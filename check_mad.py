#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:36:41 2020

@author: qingn
"""

import xarray as xr
import dask
import math
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
#from mpl_toolkits.basemap import Basemap, cm
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
# %%
# read in file as objext
path_exhaust = '/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc'
cpc = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraoscpc/maraos*.nc','10s')
cpc_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraos*.nc')
# cpc from 2017-10-29T00:00:00.360000000 to 2018-03-24T23:59:59.430000000
#exhaust_id = arm.read_netcdf('/Users/qingn/Desktop/NQ/exhaust_id/AAS*.nc')
co = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosco/mar*.nc','10s')
o3 = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraoso3/mar*.nc','10s')
# co from 2017-10-29T00:00:00.617000000 to 2018-03-24T19:59:59.200000000
# exhaust from 2017-10-18T23:45:06.000000000 to 2018-03-25T23:59:59.000000000
#exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
met = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201*.nc','10s')

#nav = arm.read_netcdf('/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.201*')

path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201*.nc'
uhsas = arm_read_netcdf_for_time_resolution(path_uhsas,'10s')
#%%
ws = met['wind_speed'].values
wd = met['wind_direction'].values
slf_time = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='10s')
slf_time_1sec = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='1s')
# change the time into same system
#dt = np.dtype(np.int64)
#time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
#time_id_date = time_id_date.values
#%%
exhaust_id = netCDF4.Dataset(path_exhaust)
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
exhaust_4mad02thresh = exhaust_id['exhaust_4mad02thresh']

exhaust_ds = xr.Dataset({'flag':('time',exhaust_4mad02thresh),'time':time_id_date})
exhaust_ds_full = exhaust_ds.resample(time='1s').nearest(tolerance='10s')
exhaust_ds_full_fill = exhaust_ds_full.fillna(1)
flag = exhaust_ds_full_fill['flag']
#%%
#
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

# 
    
'''
A= np.arange(0, 20.)
target = np.array([[-2, 100., 2., 2.4, 2.5, 2.6]])
print(A)
find_closest(A, target)

OUTPUT : array([[ 0, 19,  2,  2,  3,  3]])
'''

index_cpc_ml= module_ml.machine_learning(''.join(['./',datastream,'/*','20171111','*']) ,''.join(['./',datastream,'/mar*.nc'])) 
index_cpc_ml = np.array(index_cpc_ml[0])
# read in variables
#exhaust = exhaust_id['exhaust_4mad02thresh'][:]
ml_exhaust = cpc_ori.time[index_cpc_ml]

#%%
cpc_con = cpc['concentration']
#co_con = co['co_dry']
full_size = np.size(slf_time)
cpc_size = np.shape(cpc.time)[0]
o3_size = np.shape(o3.time)[0]
uhsas_size = np.shape(uhsas.time)[0]
co_size = np.shape(co.time)[0]
met_size = np.shape(met.time)[0]

wspd_name='wind_speed';wdir_name='wind_direction'

met_sp = np.append(met[wspd_name],[np.nan]*(full_size-met_size))
met_dir = np.append(met[wdir_name],[np.nan]*(full_size-met_size))
wind_indirection = met_sp*np.cos(met_dir/180.0*np.pi)

cpc_con = np.append(cpc_con,[np.nan]*(full_size-cpc_size))
co_con=co['co_dry'].where(co['qc_co_dry']<16384)[:full_size]
#co_con = co_con[:full_size]
o3_con = o3['o3'].where((o3['qc_o3']<262144))[:full_size]
#o3_con = o3_con[:full_size]
#o3_qc = o3['qc_o3'][:full_size]
#o3_con = np.ma.masked_where(o3_qc>2**18,o3_con)
uhsas_con = uhsas['concentration']
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
total_sas = np.sum(uhsas_con[:,:],axis=1)
total_sas = np.insert(total_sas,0,[np.nan]*1440)
total_sas = total_sas.values[:full_size]
acc_con = np.sum(uhsas_con[:,18:],axis=1)
acc_con = np.insert(acc_con,0,[np.nan]*1440)
acc_con = acc_con.values[:full_size]
#o3_con = np.append(co['co_dry'],[np.nan]*(-full_size+co_size))

#%%

ten_sec = pd.DataFrame()
#ten_sec = ten_sec.set_index(slf_time)
# We slice the exhaust to only include MARCUS authorized period.
A = np.where(exhaust_ds_full_fill.time ==np.datetime64(slf_time[0]))[0][0]
B = np.where(exhaust_ds_full_fill.time ==np.datetime64(slf_time[-1]))[0][0]
ten_exhaust = exhaust_ds_full_fill.flag[A:B+1]#.resample(time = '10min').sum()

ten_1 = ten_exhaust.values

k = np.add.reduceat(ten_1[:],np.arange(0, len(ten_1[:]),10))#[:-2]
#k = np.insert(k,0,np.sum(ten_1[:447]))
#ee = k>50
#ee = ee.astype(int)

ten_sec['flag'] = k
ten_sec['mad_contamination'] = k>3
ten_sec['index'] = range(len(ten_sec))
ten_sec = ten_sec.set_index(slf_time)
ten_sec['cpc_con'] = cpc_con
ten_sec['co_con'] = co_con
ten_sec['o3_con'] = o3_con
ten_sec['wind_indirection'] = wind_indirection
#ten_sec['qc_o3'] = o3_qc

ten_sec['total_sas'] = total_sas
ten_sec['acc_con'] = acc_con
#%% Missing cpc causes issues
new_flag = ten_sec['mad_contamination']
a = np.where(ten_sec['cpc_con']==0)[0]
print('how many flag has been turned on because of the cpc_missing(0.0_10min)', np.size(np.where(new_flag[a]==False)))
new_flag[a]=True


b = np.isnan(ten_sec['cpc_con'])
b_index = np.where(b==True)[0]
print('how many flag has been turned on because of the cpc_missing(nan_10min)', np.size(np.where(new_flag[b_index]==False)))
new_flag[b]=True
ten_sec['new_flag'] = new_flag
#%% FIGURE

formatter = DateFormatter('%m-%d %H:')
list_time = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='5D')
for i in range(len(list_time)-1):#len(list_time)-1
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4,1,figsize = (12,16),sharex =True)
#    clean_flag = np.array([~ten_sec['mad_contamination'][list_time[i]:list_time[i+1]]])[0]
    clean_flag = np.array([~ten_sec['new_flag'][list_time[i]:list_time[i+1]]])[0]
    
    ax1.xaxis.set_major_formatter(formatter); 
    ax1.plot(ten_sec[list_time[i]:list_time[i+1]].index,
             ten_sec[list_time[i]:list_time[i+1]]['acc_con'],label = 'accumulation_all',marker = '.',color = 'pink')
    ax1.plot(ten_sec[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_sec[list_time[i]:list_time[i+1]]['acc_con'][clean_flag],label = 'accu_non_stack',marker = '.',color = 'blue')

    ax1.plot(ten_sec[list_time[i]:list_time[i+1]].index,
             ten_sec[list_time[i]:list_time[i+1]]['cpc_con'],label = 'CN',marker = '.',color = 'orange')
    ax1.plot(ten_sec[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_sec[list_time[i]:list_time[i+1]]['cpc_con'][clean_flag],label = 'CN_cl',marker = '.',color = 'black')
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    ax1.set_title('contamination control',fontsize=26)
    ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
    ax1.set_ylim([0.1,2000])
    ax1.legend()
    
    
    ax2.xaxis.set_major_formatter(formatter); 
    ax2.plot(ten_sec[list_time[i]:list_time[i+1]].index,
             ten_sec[list_time[i]:list_time[i+1]]['co_con'],label = 'co',marker = '.',color = 'pink')
    ax2.plot(ten_sec[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_sec[list_time[i]:list_time[i+1]]['co_con'][clean_flag],label = 'co_cl',marker = '.',color = 'blue')
    ax2.set_ylabel('mixing ratio (ppmv)',fontsize=26,color = 'blue')
    ax2.set_ylim([0.02,0.4])
    ax2.tick_params(axis = 'y',labelcolor = 'blue')
    ax2.legend()
    ax2.yaxis.grid()
    ax2.xaxis.grid()
    
#    ax2_1 = ax2.twinx()
#    color_1 = 'tab:black'
    ax3.set_ylabel('o3_con(ppb)',color = 'black')
    ax3.plot(ten_sec[list_time[i]:list_time[i+1]].index,
             ten_sec[list_time[i]:list_time[i+1]]['o3_con'],label = 'o3',marker = '.',color = 'orange')
    ax3.plot(ten_sec[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_sec[list_time[i]:list_time[i+1]]['o3_con'][clean_flag],label = 'o3_cl',marker = '.',color = 'black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.set_ylabel('o3_con(ppb)',fontsize=26,color = 'black')
    ax3.set_ylim([-1,70])
    ax3.yaxis.grid()
    ax3.xaxis.grid()
#    ax2.set_title('contamination control',fontsize=26)

    
    ax3.legend(framealpha=0.3,loc='best')


    ax4.set_ylabel('wind_indir(m/s)',color = 'black')
    ax4.plot(ten_sec[list_time[i]:list_time[i+1]].index,
             ten_sec[list_time[i]:list_time[i+1]]['wind_indirection'],label = 'wind',marker = '.',color = 'pink')
    ax4.plot(ten_sec[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_sec[list_time[i]:list_time[i+1]]['wind_indirection'][clean_flag],label = 'wind_cl',marker = '.',color = 'blue')
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.set_ylabel('wind_indirec(m/s)',fontsize=26,color = 'blue')
    ax4.yaxis.grid()
    ax4.xaxis.grid()
    ax4.axhline(y=0,color='red',linestyle='--')
#    ax2.set_title('contamination control',fontsize=26)

    
    ax4.legend(framealpha=0.3,loc='best')
    ax = plt.gca()
    ax1.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    fig.tight_layout()
#    plt.show()
    
    cwd = os.getcwd()
    fdir=cwd+'/'+'ylim_accu_cn/'
#    fdir=cwd+'/'+'voyage_accu_clean_ccn_3ss/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'ylim_accu_cn'+'_'+str(list_time[i])+'.png')
    
    plt.gcf()
#    plt.savefig(fdir+'ylim_accu_cn'+'_'+str(list_time[i])+'.png')
    plt.savefig(fdir+'new_ylim_accu_cn'+'_'+str(list_time[i])+'.png')
    #ten_sec[list_time[i]].index
#%% Four voyages
list_date_four = ['2017-10-29','2017-12-03','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-25']
for i in range(len(list_date_four)-1):#len(list_date_four)-1
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4,1,figsize = (12,16),sharex =True)
    clean_flag = np.array([~ten_sec['new_flag'][list_date_four[i]:list_date_four[i+1]]])[0]
    
    ax1.xaxis.set_major_formatter(formatter); 
    ax1.plot(ten_sec[list_date_four[i]:list_date_four[i+1]].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['acc_con'],label = 'accumulation_all',marker = '.',color = 'pink')
    ax1.plot(ten_sec[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['acc_con'][clean_flag],label = 'accu_non_stack',marker = '.',color = 'blue')

    ax1.plot(ten_sec[list_date_four[i]:list_date_four[i+1]].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['cpc_con'],label = 'CN',marker = '.',color = 'orange')
    ax1.plot(ten_sec[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['cpc_con'][clean_flag],label = 'CN_cl',marker = '.',color = 'black')
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    ax1.set_title('contamination control',fontsize=26)
    ax1.set_ylabel('cpc (1/cm^3)',fontsize=26)
    ax1.set_ylim([0.1,2000])
    ax1.legend()
    
    
    ax2.xaxis.set_major_formatter(formatter); 
    ax2.plot(ten_sec[list_date_four[i]:list_date_four[i+1]].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['co_con'],label = 'co',marker = '.',color = 'pink')
    ax2.plot(ten_sec[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['co_con'][clean_flag],label = 'co_cl',marker = '.',color = 'blue')
    ax2.set_ylabel('mixing ratio (ppmv)',fontsize=26,color = 'blue')
    ax2.set_ylim([0.02,0.4])
    ax2.tick_params(axis = 'y',labelcolor = 'blue')
    ax2.legend()
    ax2.yaxis.grid()
    ax2.xaxis.grid()
    
#    ax2_1 = ax2.twinx()
#    color_1 = 'tab:black'
    ax3.set_ylabel('o3_con(ppb)',color = 'black')
    ax3.plot(ten_sec[list_date_four[i]:list_date_four[i+1]].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['o3_con'],label = 'o3',marker = '.',color = 'orange')
    ax3.plot(ten_sec[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['o3_con'][clean_flag],label = 'o3_cl',marker = '.',color = 'black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.set_ylabel('o3_con(ppb)',fontsize=26,color = 'black')
    ax3.set_ylim([-1,70])
    ax3.yaxis.grid()
    ax3.xaxis.grid()
#    ax2.set_title('contamination control',fontsize=26)

    
    ax3.legend(framealpha=0.3,loc='best')


    ax4.set_ylabel('wind_indir(m/s)',color = 'black')
    ax4.plot(ten_sec[list_date_four[i]:list_date_four[i+1]].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['wind_indirection'],label = 'wind',marker = '.',color = 'pink')
    ax4.plot(ten_sec[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
             ten_sec[list_date_four[i]:list_date_four[i+1]]['wind_indirection'][clean_flag],label = 'wind_cl',marker = '.',color = 'blue')
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.set_ylabel('wind_indirec(m/s)',fontsize=26,color = 'blue')
    ax4.yaxis.grid()
    ax4.xaxis.grid()
    ax4.axhline(y=0,color='red',linestyle='--')
#    ax2.set_title('contamination control',fontsize=26)

    
    ax4.legend(framealpha=0.3,loc='best')
    ax = plt.gca()
    ax1.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    fig.tight_layout()
#    plt.show()
    
    cwd = os.getcwd()
    fdir=cwd+'/'+'voyage_accu_cn/'
#    fdir=cwd+'/'+'voyage_accu_clean_ccn_3ss/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'voyage_accu_cn'+'_'+str(list_date_four[i])+'.png')
    
    plt.gcf()
#    plt.savefig(fdir+'voyage_accu_cn'+'_'+str(list_date_four[i])+'.png')
    
    #%%
aerosol_df = ten_sec[['index','new_flag','mad_contamination','cpc_con']]
ten_sec.to_csv('uhsas_gas_ten_10min_after_missingcpc_0_nan.csv', index=True)
aerosol_df.to_csv('ten_10min_after_missingcpc_0_nan.csv', index=True)
#%%
a = ten_sec['2018-03-10':'2018-03-12']

#%% Figure original cpc
#plt.figure(figsize = (12,16),)
#formatter = DateFormatter('%m-%d %H:')
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (12,16),sharex =True)
ax1.plot(ten_sec['cpc_con']['2017-11'],'.b')
ax1.plot(ten_sec['cpc_con'][np.where()])
ax2.plot(ten_sec['co_con']['2017-11'],'.b')
ax1.set_ylim(bottom=0.0)
ax2.set_ylim(bottom=0.0,top = 15.0)
ax1.set_ylabel('cpc concentration(#/cc)',fontsize=26,color = 'blue')
ax2.set_ylabel('co mixing ratio(ppmv)',fontsize=26,color = 'blue')
fig.autofmt_xdate()
ax1.set_xlabel('time')
fig.tight_layout()
plt.title('original_cpc_co')

#ax3.plot(ten_sec['c']['2017-11'],'.b')

fdir='/Users/qingn/Documents/course_qing/BUL_2020_spring/'+'seminar_figure/'

#    fdir=cwd+'/'+'voyage_accu_clean_ccn_3ss/'
try:
    os.stat(fdir)
except:
    os.mkdir(fdir)
print('original_cpc_co')
#plt.savefig(fdir+'original_cpc_co'+'.png')
#%% Compare UHSAS and CPC
plt.figure(figsize=[16,4])
formatter = DateFormatter('%m-%d')
plt.plot(ten_sec['cpc_con']['2017-11'],'.',label = 'CPC_CN')
plt.plot(ten_sec['total_sas']['2017-11'],'*',alpha=0.1,label='UHSAS')
plt.ylim(bottom =0.1)
ax= plt.gca();ax.xaxis.set_major_formatter(formatter)
ax.set_yscale('log')

plt.legend()
plt.ylabel('concentration(cc-1)')
plt.xlabel('time')
plt.title('Compare UHSAS and CPC')
#fig.autofmt_xdate()
fig.tight_layout()
#plt.show()
#%% Machine Learning training data
plt.figure(figsize=[16,4])
plt.plot(ten_sec['2017-11-11':'2017-11-11']['co_con'])
fig.tight_layout()
plt.xlabel('time')
plt.ylabel('CN concentration(cc-1)')
formatter = DateFormatter('%m-%d %H:')
ax= plt.gca();ax.xaxis.set_major_formatter(formatter)
