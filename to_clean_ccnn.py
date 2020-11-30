#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:10:05 2020

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
#matplotlib.use('Agg')
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
FONTSIZE = 26
LABELSIZE = 26
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
#%%
path_exhaust = '/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc'
path_ccnspectra1 = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201*.nc'
ccn_col = arm.read_netcdf('/Users/qingn/Desktop/NQ/marccn1col1/maraosccn1colM1.b1.201710*.nc')
ccn_spectra = arm_read_netcdf(path_ccnspectra1)
#ccn_avg = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1.20180324*')
ccn_avg = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1.20171029*')
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201*.nc'
uhsas = arm_read_netcdf_for_time_resolution(path_uhsas,'10s') # ten seconds resolution


#%%
exhaust_id = netCDF4.Dataset(path_exhaust)
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
exhaust_4mad02thresh = exhaust_id['exhaust_4mad02thresh']

slf_time = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='s')
slf_time_ten = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='10min')
#slf_time1 = pd.date_range(start ='2017-10-18T23:45:06',end = '2018-03-25T23:59:59', freq='s')
#flag = [0]*np.size(slf_time)
#mask = slf_time.isin(time_id_date)

exhaust_ds = xr.Dataset({'flag':('time',exhaust_4mad02thresh),'time':time_id_date})
exhaust_ds_full = exhaust_ds.resample(time='1s').nearest(tolerance='10s')
exhaust_ds_full_fill = exhaust_ds_full.fillna(1)
flag = exhaust_ds_full_fill['flag']

#%%
ten_min = pd.DataFrame()

spectra= ccn_spectra['N_CCN'].values
spectra_list = np.reshape(spectra,3517*6)
qc = np.reshape(ccn_spectra['qc_N_CCN'].values,3517*6)
ten_min['N_CCN'] = spectra_list[:-5] # since spectra has after 3-24 12pm, slice those out
set_point = [0,.1,.2,.5,.8,1.0]
ss = set_point*(3517)
ten_min['SS'] = ss[:-5]
ten_min['qc_N_CCN'] = qc[:-5]
ten_min = ten_min.set_index(slf_time_ten)
# We slice the exhaust to only include MARCUS authorized period.
A = np.where(exhaust_ds_full_fill.time ==np.datetime64(slf_time[0]))[0][0]
B = np.where(exhaust_ds_full_fill.time ==np.datetime64(slf_time[-1]))[0][0]
ten_exhaust = exhaust_ds_full_fill.flag[A:B+1]#.resample(time = '10min').sum()
#%%
ten_1 = ten_exhaust.values

#ten_min['flag'][0] = np.sum(ten_1[:447])
k = np.add.reduceat(ten_1[448:],np.arange(0, len(ten_1[448:]),600))#[:-2]
k = np.insert(k,0,np.sum(ten_1[:447]))
#ee = k>50
#ee = ee.astype(int)

ten_min['flag'] = k
ten_min['contamination'] = k>50
ten_min['index'] = range(len(ten_min))
#ten_min['time'] = ten_exhaust.time

#%%
con = uhsas['concentration'] 

#interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
acc_con = np.sum(con[:,18:],axis=1)
uhsas_con = np.sum(con,axis=1)
sas_60_100 = np.sum(con[:,:18],axis=1)
sas_100_350 = np.sum(con[:,18:61],axis=1)
sas_350_ = np.sum(con[:,61:],axis=1)

con_10min = np.add.reduceat(con[:-1080,:].values,np.arange(0, len(uhsas_con[:-1080]),60))
con_10min = np.insert(con_10min,[0], empty,axis =0)

#psd = con_10min/60.

ten_min['time'] = ten_min.index
ten_min_array = ten_min.to_xarray()
ten_min_array = ten_min_array.set_coords('time')
ten_min_array = ten_min_array.swap_dims({"index": "time"})
ten_min_array['upper_size_limit'] = uhsas['upper_size_limit'][0,]
ten_min_array = ten_min_array.set_coords('upper_size_limit')
ten_min_array['psd'] = (con_10min/60.)


uhsas_10min = np.add.reduceat(uhsas_con[:-1080].values,np.arange(0, len(uhsas_con[:-1080]),60))#[:-2]
uhsas_10min = np.insert(uhsas_10min,0,[np.nan]*25)
ten_min['uhsas'] = uhsas_10min/60.
#ten_min['uhsas'] = uhsas_10min

sas_60_100_10min = np.add.reduceat(sas_60_100[:-1080].values,np.arange(0, len(sas_60_100[:-1080]),60))#[:-2]
sas_60_100_10min = np.insert(sas_60_100_10min,0,[np.nan]*25)
ten_min['uhsas_small'] = sas_60_100_10min/60.
#ten_min['uhsas'] = uhsas_10min
sas_100_350_10min = np.add.reduceat(sas_100_350[:-1080].values,np.arange(0, len(sas_60_100[:-1080]),60))#[:-2]
sas_100_350_10min = np.insert(sas_100_350_10min,0,[np.nan]*25)
ten_min['uhsas_middle'] = sas_100_350_10min/60.

sas_350_10min = np.add.reduceat(sas_350_[:-1080].values,np.arange(0, len(sas_350_[:-1080]),60))#[:-2]
sas_350_10min = np.insert(sas_350_10min,0,[np.nan]*25)
ten_min['uhsas_large'] = sas_350_10min/60.
#%%
acc_10min = np.add.reduceat(acc_con[:-1080].values,np.arange(0, len(acc_con[:-1080]),60))#[:-2]
acc_10min = np.insert(acc_10min,0,[np.nan]*25)
ten_min['accumulation'] = acc_10min/60.


ten_min_newflag = pd.read_csv("/Users/qingn/Desktop/NQ/ten_10min_after_missingcpc_0_nan.csv")

flag_int = list(map(int,ten_min_newflag['new_flag'].values))
flag_10min = np.add.reduceat(flag_int,np.arange(0, len(flag_int),60))#[:-2]
ten_min['ship_stack'] = flag_10min
ten_min['new_flag'] = flag_10min <10
print('how many ship off in total', np.size(np.where(ten_min['new_flag']==0)[0]))


#%%

b = ten_min['accumulation'].values==0
ten_min['accumulation'][b]=np.nan
#%%
ten_min['uhsas_small'][b]=np.nan
ten_min['uhsas_middle'][b]=np.nan
ten_min['uhsas_large'][b]=np.nan
#%% Plotting general clean CCN
#formatter = DateFormatter('%H:%M')
#%H:%M
formatter = DateFormatter('%d_%H:')

for i in range(6):
    fig=plt.figure(figsize=(FIGWIDTH*3,FIGHEIGHT))
    plt.plot(ten_min[ten_min['SS'].values==set_point[i]].index,
             ten_min['N_CCN'][ten_min['SS'].values==set_point[i]],'b.',label = 'ori')
    #plt.plot_date(ten_min[ten_min['SS'].values==0.5].index,ten_min['N_CCN'][ten_min['SS'].values==0.5],'b.',label = 'ori')
    #plt.plot(ten_min[ten_min['SS'].values==set_point[2]][~ten_min['contamination']].index,ten_min['N_CCN'][ten_min['SS'].values==set_point[2]][~ten_min['contamination']],'g.',label = 'non_stack')
    plt.plot(ten_min[ten_min['SS'].values==set_point[i]][~ten_min['contamination']][ten_min['qc_N_CCN']==0].index,
             ten_min['N_CCN'][ten_min['SS'].values==set_point[i]][~ten_min['contamination']][ten_min['qc_N_CCN']==0],'y.',label = 'bit_clean_non_stack')
    #plt.plot_date(ten_min[ten_min['SS'].values==0.5][ten_min['flag']==0].index,ten_min['N_CCN'][ten_min['SS'].values==0.5][ten_min['flag']==0],'r.',label = 'stack')
    #plt.plot(ten_min.all([ten_min['SS'].values==0.5] , [ten_min['flag'].values==0]).index, ten_min['N_CCN'].all([ten_min['SS'].values==0.5 , ten_min['flag'].values==0]))
    ax = plt.gca()
    #ax.xaxis.set_major_locator(loc)
    #ax.xaxis.set_major_formatter(formatter)
    plt.title('NCCN_'+str(set_point[i])+'%')
    ax.set_xlabel('time(10 mins)')
    ax.set_ylabel('N_CCN(1/cm^3)')
    ax.legend()
    fig.tight_layout()
    cwd = os.getcwd()
    fdir=cwd+'/'+'clean_ccn_ss/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'ccn'+'_'+str(i)+'.png')
    
    plt.gcf()
#    plt.savefig(fdir+'ccn'+'_'+str(i)+'.png')  

#%% figures to show how the ship stack flag counts
plt.figure(figsize = [18,4])
plt.plot(ccn_col.time[:5000],ccn_col.seconds_after_transition[:5000])
plt.plot(ccn_avg.time[:8],[0]*8,'r.')
plt.title('seconds_after_transition')
plt.xlabel('time')
plt.ylabel('seconds')
#%%
formatter = DateFormatter('%H:%M')
fig = plt.figure(figsize = [16,9])
plt.plot(ccn_col.time[:5000],ccn_col.supersaturation_set_point[:5000],label = 'ss')
plt.plot(ccn_avg.time[:8],[0]*8,'r.',markersize=12,label = 'avg_measurement')
ax= plt.gca()
ax.xaxis.set_major_formatter(formatter)
plt.title('supersatuation set point cycle with time',fontsize = 34)
plt.xlabel('time',fontsize = 34)

plt.ylabel('super_saturation(%)')
fig.tight_layout()
plt.legend()

print(ccn_avg.time[:8])
#np.where(ccn_col.seconds_after_transition[:4000].values==599)
#%%
date_list = ['2017-10-29','2018-01-01','2017-11-11']
A = date_list[0]
B = date_list[2]
for i in [1,2,3,4,5]:
    fig=plt.figure(figsize=(FIGWIDTH*3,FIGHEIGHT))
#    plt.plot(ten_min['2017-10-29'].accumulation)
#    plt.plot(ten_min['2017-10-29'][])
    plt.plot(ten_min[ten_min['SS']==set_point[i]][A:B].index,
             ten_min['N_CCN'][ten_min['SS']==set_point[i]][A:B],'b.',label = 'ori')
    
#    plt.plot(ten_min[ten_min['SS']==set_point[i]].index,
#             ten_min['N_CCN'][ten_min['SS']==set_point[i]],'b.',label = 'ori')
    plt.plot(ten_min[ten_min['SS']==set_point[i]][~ten_min['contamination']][ten_min['qc_N_CCN']==0][A:B].index,
             ten_min['N_CCN'][ten_min['SS']==set_point[i]][~ten_min['contamination']][ten_min['qc_N_CCN']==0][A:B],'y.',label = 'bit_clean_non_stack')
#
#    plt.plot(ten_min[ten_min['SS']==set_point[i]][~ten_min['contamination']][ten_min['qc_N_CCN']==0].index,
#             ten_min['N_CCN'][ten_min['SS']==set_point[i]][~ten_min['contamination']][ten_min['qc_N_CCN']==0],'y.',label = 'bit_clean_non_stack')
    ax = plt.gca()
    #ax.xaxis.set_major_locator(loc)
    #ax.xaxis.set_major_formatter(formatter)
    plt.title('NCCN_'+str(set_point[i])+'%')
    ax.set_xlabel('time(10 mins)')
    ax.set_ylabel('N_CCN(1/cm^3)')
    ax.legend()
    
#%% For Four voages
formatter = DateFormatter('%m-%d %H:')
list_date_four = ['2017-10-29','2017-12-03','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-25']
for i in range(len(list_date_four)-1):#len(list_date_four)-1
#    print(list_date_four[i]+':'+list_date_four[i+1])
#    [~ten_min['contamination']]
#    clean_flag = np.array([~ten_min['contamination'][list_date_four[i]:list_date_four[i+1]]])[0]
    clean_flag = np.array([ten_min['new_flag'][list_date_four[i]:list_date_four[i+1]]])[0]
    clean_bit = np.array([ten_min['qc_N_CCN'][list_date_four[i]:list_date_four[i+1]]==0])[0]
    supers_1 = np.array([ten_min['SS'][list_date_four[i]:list_date_four[i+1]]==0.1])[0]
    supers_2 = np.array([ten_min['SS'][list_date_four[i]:list_date_four[i+1]]==0.2])[0]
    supers_5 = np.array([ten_min['SS'][list_date_four[i]:list_date_four[i+1]]==0.5])[0]
    
    
#    
#    flag = ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][1][2]
#    trytry = ten_min['qc_N_CCN'][list_date_four[i]:list_date_four[i+1]]==0
    fig=plt.figure(figsize=(FIGWIDTH*2.3,FIGHEIGHT*1.2))
    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]].index,
             ten_min[list_date_four[i]:list_date_four[i+1]]['accumulation'],label = 'accumulation_all',marker = '.',color = 'pink')
    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
             ten_min[list_date_four[i]:list_date_four[i+1]]['accumulation'][clean_flag],label = 'accu_no-stack',marker = '.',color = 'blue')


#''' 
#Clean accumulation and unclean ccn 
#    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][clean_flag].index,
#             ten_min[list_date_four[i]:list_date_four[i+1]]['accumulation'][clean_flag],label = 'accumulation_cl',marker = '.')
#
#
#    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][ten_min['SS']==0.1].index,
#             ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][ten_min['SS']==0.1],label = '1%')
#    
#    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][ten_min['SS']==0.2].index,
#             ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][ten_min['SS']==0.2],label = '2%')
#    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][ten_min['SS']==0.5].index,
#             ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][ten_min['SS']==0.5],label = '5%')
#    
#'''
#
# Clean accumulation and clean ccn
    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][(clean_flag)&(clean_bit)&(supers_1)].index,
             ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_1)],'y.',label = '.1%',)
    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][(clean_flag)&(clean_bit)&(supers_2)].index,
             ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_2)],'g.',label = '.2%')
    plt.plot(ten_min[list_date_four[i]:list_date_four[i+1]][(clean_flag)&(clean_bit)&(supers_5)].index,
         ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_5)],'r.',label = '.5%')
    plt.ylim(-1,600)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    
    num_clean = np.sum(clean_flag)
    num_total = np.size(clean_flag)
    num_1 =  np.size(ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_1)])
    num_2 =  np.size(ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_2)])
    num_5 =  np.size(ten_min[list_date_four[i]:list_date_four[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_5)])
# TEXT
    ax.text(0.8, 0.55, 'ss=0.1 #: %d'%(num_1),transform=ax.transAxes)
    ax.text(0.8, 0.65, 'ss = 0.2 #: %d'%(num_2),transform=ax.transAxes)
    ax.text(0.8, 0.75, 'ss = 0.5 #: %d'%(num_5),transform=ax.transAxes)
    ax.text(0.8, 0.355, 'clean #: %03d'%(num_clean),transform=ax.transAxes)
    ax.text(0.8, 0.45, 'total #: %03d'%(num_total),transform=ax.transAxes)
    plt.legend(framealpha=0.3,loc='upper left')
#    plt.legend()
    plt.ylabel('concentration(1/cc)')
    plt.xlabel('time')
    plt.title('Clean_CCN and un/clean_accumulation mode')
    fig.autofmt_xdate()
    fig.tight_layout()

    cwd = os.getcwd()
    fdir=cwd+'/'+'ylim_voyage_accu_clean_ccn_3ss/'
#    fdir=cwd+'/'+'voyage_accu_clean_ccn_3ss/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'accu_ccn_3ss'+'_'+str(i)+'.png')
    
    plt.gcf()
    plt.savefig(fdir+'ylim_voage_accu_clean_ccn_3ss'+'_'+str(i)+'.png')
#    plt.savefig(fdir+'voyage_accu_clean_ccn_3ss'+'_'+str(i)+'.png')  
    
#%%
formatter = DateFormatter('%m-%d %H:')
list_time = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='5D')

for i in range(len(list_time)-1):#len(list_time)-1
#    print(list_time[i]+':'+list_time[i+1])
#    [~ten_min['contamination']]
#    clean_flag = np.array([~ten_min['contamination'][list_time[i]:list_time[i+1]]])[0]
    clean_flag = np.array([ten_min['new_flag'][list_time[i]:list_time[i+1]]])[0]
    clean_bit = np.array([ten_min['qc_N_CCN'][list_time[i]:list_time[i+1]]==0])[0]
    supers_1 = np.array([ten_min['SS'][list_time[i]:list_time[i+1]]==0.1])[0]
    supers_2 = np.array([ten_min['SS'][list_time[i]:list_time[i+1]]==0.2])[0]
    supers_5 = np.array([ten_min['SS'][list_time[i]:list_time[i+1]]==0.5])[0]
    
    
#    
#    flag = ten_min[list_time[i]:list_time[i+1]]['N_CCN'][1][2]
#    trytry = ten_min['qc_N_CCN'][list_time[i]:list_time[i+1]]==0
    fig=plt.figure(figsize=(FIGWIDTH*2,FIGHEIGHT))
    plt.plot(ten_min[list_time[i]:list_time[i+1]].index,
             ten_min[list_time[i]:list_time[i+1]]['accumulation'],label = 'accumulation_mode',marker = '.',color = 'pink')
    plt.plot(ten_min[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_min[list_time[i]:list_time[i+1]]['accumulation'][clean_flag],label = 'accumulation_cl',marker = '.',color = 'blue')


#''' 
#Clean accumulation and unclean ccn 
#    plt.plot(ten_min[list_time[i]:list_time[i+1]][clean_flag].index,
#             ten_min[list_time[i]:list_time[i+1]]['accumulation'][clean_flag],label = 'accumulation_cl',marker = '.')
#
#
#    plt.plot(ten_min[list_time[i]:list_time[i+1]][ten_min['SS']==0.1].index,
#             ten_min[list_time[i]:list_time[i+1]]['N_CCN'][ten_min['SS']==0.1],label = '1%')
#    
#    plt.plot(ten_min[list_time[i]:list_time[i+1]][ten_min['SS']==0.2].index,
#             ten_min[list_time[i]:list_time[i+1]]['N_CCN'][ten_min['SS']==0.2],label = '2%')
#    plt.plot(ten_min[list_time[i]:list_time[i+1]][ten_min['SS']==0.5].index,
#             ten_min[list_time[i]:list_time[i+1]]['N_CCN'][ten_min['SS']==0.5],label = '5%')
#    
#'''
#
# Clean accumulation and clean ccn
    plt.plot(ten_min[list_time[i]:list_time[i+1]][(clean_flag)&(clean_bit)&(supers_1)].index,
             ten_min[list_time[i]:list_time[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_1)],'y.',label = '.1%',)
    plt.plot(ten_min[list_time[i]:list_time[i+1]][(clean_flag)&(clean_bit)&(supers_2)].index,
             ten_min[list_time[i]:list_time[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_2)],'g.',label = '.2%')
    plt.plot(ten_min[list_time[i]:list_time[i+1]][(clean_flag)&(clean_bit)&(supers_5)].index,
         ten_min[list_time[i]:list_time[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_5)],'r.',label = '.5%')
    plt.ylim(0,600)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    
    num_clean = np.sum(clean_flag)
    num_total = np.size(clean_flag)
    num_1 =  np.size(ten_min[list_time[i]:list_time[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_1)])
    num_2 =  np.size(ten_min[list_time[i]:list_time[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_2)])
    num_5 =  np.size(ten_min[list_time[i]:list_time[i+1]]['N_CCN'][(clean_flag)&(clean_bit)&(supers_5)])
# TEXT
    ax.text(0.5, 0.55, 'ss=0.1 #: %d'%(num_1),transform=ax.transAxes)
    ax.text(0.5, 0.65, 'ss = 0.2 #: %d'%(num_2),transform=ax.transAxes)
    ax.text(0.5, 0.75, 'ss = 0.5 #: %d'%(num_5),transform=ax.transAxes)
    ax.text(0.5, 0.355, 'clean #: %d'%(num_clean),transform=ax.transAxes)
    ax.text(0.5, 0.45, 'total #: %03d'%(num_total),transform=ax.transAxes)
    plt.legend(framealpha=0.3,loc='upper left')
#    plt.legend()
    plt.xlabel('concentration(1/cc)')
    plt.ylabel('time')
    plt.title('Clean_CCN and un/clean_accumulation mode')
    fig.tight_layout()
    cwd = os.getcwd()
    fdir=cwd+'/'+'ylim_accu_clean_ccn_3ss/'
#    fdir=cwd+'/'+'accu_clean_ccn_3ss/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'accu_ccn_3ss'+'_'+str(i)+'.png')
    
    plt.gcf()
    plt.savefig(fdir+'new_ylim_accu_clean_ccn_3ss'+'_'+str(i)+'.png')
#    plt.savefig(fdir+'accu_clean_ccn_3ss'+'_'+str(i)+'.png')
#%% Statistical analysis for the ten_min
    
    
full_clean_flag = np.array([~ten_min['contamination']])[0]
full_clean_bit = np.array(ten_min['qc_N_CCN']==0)[0]
full_supers_1 = np.array([ten_min['SS']==0.1])[0]
full_supers_2 = np.array([ten_min['SS']==0.2])[0]
full_supers_5 = np.array([ten_min['SS']==0.5])[0]
full_supers_8 = np.array([ten_min['SS']==0.8])[0]
full_supers_10 = np.array([ten_min['SS']==1.0])[0]
#%%
clean_df_1 = ten_min[(full_clean_flag)&(full_clean_bit)&(full_supers_1)]
clean_df_2 = ten_min[(full_clean_flag)&(full_clean_bit)&(full_supers_2)]
clean_df_5 = ten_min[(full_clean_flag)&(full_clean_bit)&(full_supers_5)]
clean_df_8 = ten_min[(full_clean_flag)&(full_clean_bit)&(full_supers_8)]
clean_df_10 = ten_min[(full_clean_flag)&(full_clean_bit)&(full_supers_10)]
print('how many clean ss=0.1% ccn left?',np.shape(clean_df_1)[0])
print('how many clean ss=0.2% ccn left?',np.shape(clean_df_2)[0])
print('how many clean ss=0.5% ccn left?',np.shape(clean_df_5)[0])
print('how many clean ss=0.8% ccn left?',np.shape(clean_df_8)[0])
print('how many clean ss=1.0% ccn left?',np.shape(clean_df_10)[0])
#np.shape(clean_df_2)[0]
#np.shape(clean_df_5)[0]

#%% Figures include the original and clean acuumulation mode aerosol and statistical analysis in each period
formatter = DateFormatter('%m-%d %H:')
list_time = pd.date_range(start ='2017-10-29T00:00:00',end = '2018-03-24T12:00:00', freq='5D')
for i in range(len(list_time)-1):#len(list_time)-1
    clean_flag = np.array([~ten_min['contamination'][list_time[i]:list_time[i+1]]])[0]
    fig=plt.figure(figsize=(FIGWIDTH*2,FIGHEIGHT),dpi=70)
    plt.plot(ten_min[list_time[i]:list_time[i+1]].index,
             ten_min[list_time[i]:list_time[i+1]]['accumulation'],label = 'accumulation_mode',marker = '.',color = 'orange')

    plt.plot(ten_min[list_time[i]:list_time[i+1]][clean_flag].index,
             ten_min[list_time[i]:list_time[i+1]]['accumulation'][clean_flag],label = 'accumulation_clean',marker = '.')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    num_clean = np.sum(clean_flag)
    num_total = np.size(clean_flag)
#    print(np.shape(clean_flag)[0])
#    ax.axis([0,10,0,10])
    ax.text(0.5, 0.55, 'clean counts: %03d'%(num_clean),transform=ax.transAxes)
    ax.text(0.5, 0.65, 'total counts: %03d'%(num_total),transform=ax.transAxes)
#    plt.text(0.5, 0.1, np.shape(clean_flag)[0], fontsize=FONTSIZE,horizontalalignment='left',
#         verticalalignment='center',)
#    text(0.5, 0.5, 'matplotlib', horizontalalignment='center',
#             verticalalignment='center', transform=ax.transAxes)
    plt.legend(framealpha=0.5)
    plt.ylabel('concentration(1/cc)')
    plt.xlabel('time')
    plt.title('Clean_accumulation mode')
    fig.tight_layout()
    
    cwd = os.getcwd()
    fdir=cwd+'/'+'clean_accu/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    print('Writing: '+fdir+'accu'+'_'+str(i)+'.png')
    
    plt.gcf()
    plt.savefig(fdir+'accu'+'_'+str(i)+'.png')

#%% SAVE dataframe
    
ten_min.to_csv('ten_min_clean_cpc_aftms_accu.csv', index=True)
   


