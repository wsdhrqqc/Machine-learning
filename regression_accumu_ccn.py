#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:09:50 2019
 accumulation mode contribute to CCN
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
import scipy
from act.io.armfiles import read_netcdf
from scipy import stats
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
FONTSIZE = 22
LABELSIZE = 22
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['axes.labelsize']=FONTSIZE
plt.rcParams['axes.titlesize']=FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams["legend.framealpha"] = 0.3

matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE) 

HOME_DIR = str(pathlib.Path.home()/'Desktop/NQ' )


def arm_read_netcdf(full_directory_filebase,sdate,edate):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    files = glob.glob(full_directory_filebase)
    files.sort()
    files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]   
#    file_dir = str(HOME_DIR + directory_filebase)
    file_ori = arm.read_netcdf(files)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
    
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time='1h').nearest()
#    file = file_ori.resample(time='1h').std()
    return file

def arm_read_netcdf2(full_directory_filebase,sdate,edate):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    files = glob.glob(full_directory_filebase)
    files.sort()
    files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]   
#    file_dir = str(HOME_DIR + directory_filebase)
    file_ori = arm.read_netcdf(files)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
    
    file = file_ori
#resample(time='10s').mean()
#    file = file_ori.resample(time='1h').nearest()
    return file
#%%
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
#%%
sdate='20171126'
edate='20171129'
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201711**.nc'
uhsas = arm_read_netcdf2(path_uhsas,sdate,edate)
uhsas2 = arm_read_netcdf(path_uhsas,sdate,edate)
uh_con = uhsas['concentration'][:,19:].sum(axis=1)
uh_con2 = uhsas2['concentration'][:,19:].sum(axis=1)
dataframe_uhsas = pd.DataFrame({'uhsas':uh_con},index = uh_con.time.values)
dataframe_uhsas.index.name = 'time'
#dataframe_uhsas = dataframe_uhsas.set_index(dataframe_uhsas['time'])
uh_con_std = uh_con.resample(time = "1h").std('time')
uh_con_med = uh_con.resample(time="1h").median()
b = uh_con.resample(time='1h').mean('time').std('time')
#pd.core.resample.Resampler.median(uh_con_std)
#whole_voyage = pd.read_csv('/Users/qingn/four_voyage_env_ccn.csv',parse_dates = True,index_col = 'Unnamed: 0' )

path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201711*.nc'
ccn_colavg = arm_read_netcdf(path_ccn,sdate,edate)
ccn_con1 = ccn_colavg['N_CCN']
#con = ccn_colavg['N_CCN']
#con_1 = con[:,1]
qc_con = ccn_colavg['qc_N_CCN']

#qc_con_1 = qc_con[:,1]
time_ccn = ccn_colavg['time'].values

#idx1 = find_closest(uh_con.time.values, ccn_con1.time)

#ccn_con1 = ccn_con1[idx1]
ccn_con1 = np.ma.masked_where(qc_con!= 0, ccn_con1) 


lower_bd = uhsas['lower_size_limit'].values[0,]
upper_bd = uhsas['upper_size_limit'].values[0,]
#interval = upper_bd-lower_bd
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
con_non_nan = np.nan_to_num(uh_con) # meta automatically dispeared
#total_con = np.dot(con_non_nan, interval_meta[19:])
#ccn_con = whole_voyage['2017-11-27 02':'2017-11-28 10']['ccn']

#idx2 = find_closest(uh_con.time.values, ccn_con.index)

#uhsas_con = total_con[idx2] # to create a relative smaller exhaust_id array
'''This step is slow!!'''
#plt.scatter(ccn_con,uhsas_con)
#plt.scatter(uhsas_con, ccn_con)

dataframe_ccn = pd.DataFrame({'accumulation_mode':uh_con2,'std':uh_con_std,'con_at_0.1%':ccn_con1[:,1],'con_at_0.2%':ccn_con1[:,2],'con_at_0.5%':ccn_con1[:,3],'con_at_0.8%':ccn_con1[:,4],'con_at_1.0%':ccn_con1[:,5]},index = time_ccn)
a = dataframe_ccn['2017-11-27 02':'2017-11-28 10']
#a = dataframe_ccn
#%%
print(scipy.stats.pearsonr(a['accumulation_mode'],a['con_at_0.5%']))
sns.set(color_codes=True)

asymmetric_error = [a['accumulation_mode']-a['std'], a['accumulation_mode']+a['std']]
fig = plt.figure()
ax = plt.gca
plt.errorbar(a['accumulation_mode'], a['con_at_0.5%'], yerr=asymmetric_error, fmt='o')
plt.scatter(a['accumulation_mode'], a['con_at_0.5%'])
#sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6,"xtick.major.size": 6, "xtick.major.size": 6})

rc={'axes.labelsize': 18, 'font.size': 18, 'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize': 18, 'axes.titlesize': 18}
sns.set(rc)
g = sns.jointplot(x = 'accumulation_mode',y = 'con_at_0.5%',data = a, kind = 'reg', stat_func=r2)
g.set_axis_labels('accumulation_mode(#/cc)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 20)
#%%
sdate='20180119'
edate='20180122'
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201801**.nc'
uhsas = arm_read_netcdf(path_uhsas,sdate,edate)
uh_con = uhsas['concentration'][:,19:]


whole_voyage = pd.read_csv('/Users/qingn/four_voyage_env_ccn.csv',parse_dates = True,index_col = 'Unnamed: 0' )

path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201801*.nc'
ccn_colavg = arm_read_netcdf(path_ccn,sdate,edate)
ccn_con1 = ccn_colavg['N_CCN']
#con = ccn_colavg['N_CCN']
#con_1 = con[:,1]
qc_con = ccn_colavg['qc_N_CCN']

#qc_con_1 = qc_con[:,1]
time_ccn = ccn_colavg['time'].values

#idx1 = find_closest(uh_con.time.values, ccn_con1.time)
#ccn_con1 = ccn_con1[idx1]
ccn_con1 = np.ma.masked_where(qc_con!= 0, ccn_con1) 


lower_bd = uhsas['lower_size_limit'].values[0,]
upper_bd = uhsas['upper_size_limit'].values[0,]
#interval = upper_bd-lower_bd
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
con_non_nan = np.nan_to_num(uh_con) # meta automatically dispeared
total_con = np.dot(con_non_nan, interval_meta[19:])
#ccn_con = whole_voyage['2017-11-27 02':'2017-11-28 10']['ccn']

#idx2 = find_closest(uh_con.time.values, ccn_con.index)

#uhsas_con = total_con[idx2] # to create a relative smaller exhaust_id array
'''This step is slow!!'''
#plt.scatter(ccn_con,uhsas_con)
#plt.scatter(uhsas_con, ccn_con)

dataframe_ccn_1 = pd.DataFrame({'accumulation_mode':total_con,'con_at_0.1%':ccn_con1[:,1],'con_at_0.2%':ccn_con1[:,2],'con_at_0.5%':ccn_con1[:,3],'con_at_0.8%':ccn_con1[:,4],'con_at_1.0%':ccn_con1[:,5]},index = time_ccn)
b = dataframe_ccn_1['2018-01-19 02':'2018-01-21 03'].dropna()
print(scipy.stats.pearsonr(b['accumulation_mode'],b['con_at_0.5%']))
sns.set(color_codes=True)#,xlim=[30,1000],ylim=[0.01,300]
g =sns.jointplot(x = 'accumulation_mode',y = 'con_at_0.5%', data = b, kind = 'reg')
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)
#%%
c = pd.concat([a,b])
ccn_con = whole_voyage['2017-11-27 02':'2018-01-21 03']['ccn']
idx = find_closest(c.index.values,ccn_con.index.values)
c = c.iloc[idx]

#ccn_con1 = ccn_con1[idx1]


#c = c.dropna()
print(scipy.stats.pearsonr(c['accumulation_mode'],c['con_at_0.5%']))
sns.set(color_codes=True)#,xlim=[30,1000],ylim=[0.01,300]
g =sns.jointplot(x = 'accumulation_mode',y = 'con_at_0.5%', data = c, kind = 'reg')
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)
