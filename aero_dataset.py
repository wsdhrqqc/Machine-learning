
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:30:14 2020

Create aerosol dataset

@author: qingn
"""
import xarray as xr
from pandas import Grouper
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
#    file = file_ori.resample(time='1h').nearest(tolerance = '2h')
    return file_ori

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

def arm_read_netcdf_for_time_resolution_mean(directory_filebase, time_resolution):
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
#    resample(time="1D").interpolate("linear")
#    file = file_ori.resample(time=time_resolution).interpolate()
    file = file_ori.resample(time=time_resolution).mean()
    return file
#%%
df = pd.read_csv('/Users/qingn/ten_min_clean_cpc_aftms_accu.csv', parse_dates=True)
# Flag with CCN 21097#
met = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201*.nc','1h')
#met1 = arm_read_netcdf('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.20171029*.nc')
uhsas = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201*.nc','10s')
#wind = pd.read_csv('/Users/qingn/four_voyage_env_ccn.csv')
wind_1_cpc = pd.read_csv('/Users/qingn/201711cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_2_cpc = pd.read_csv('/Users/qingn/201712cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_3_cpc = pd.read_csv('/Users/qingn/201801cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_4_cpc = pd.read_csv('/Users/qingn/201802cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_5_cpc = pd.read_csv('/Users/qingn/201803cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)

wind_full = pd.concat([wind_1_cpc,wind_2_cpc,wind_3_cpc,wind_4_cpc,wind_5_cpc])
wind_full_10min = wind_full.resample('10T').mean()
del wind_1_cpc,wind_2_cpc,wind_3_cpc,wind_4_cpc,wind_5_cpc
#%% Used for boxplot
#groups = wind_full.groupby(Grouper(freq = 'D'))
#months = pd.concat([pd.DataFrame(x[1].values) for x in groups],axis=1)
#slf_time = pd.date_range(start ='2017-10-29',end = '2018-03-24', freq='D')# All together 147 days
##months.columns  = slf_time
#error = months.std()
#err = np.array(error)
#%% We transfer 10s uhsas into 10min UHSAS


#list_date_four = ['2017-10-02','2017-12-03','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-24']
psd = uhsas['concentration'] # 1265400
psd = psd.resample(time="10T").mean(dim = 'time') # (21090, 99)shape
#lower_limit = uhsas['lower_size_limit'][0].values
pd_psd = psd.to_pandas()
pd_psd['date'] = pd_psd.index

pd_psd.to_csv('uhsas_99_con.csv',index=True)
#%% Precipitation


rain = met['rain_intensity'] # 3525 item
#rain = rain.resample(time='10T').interpolate('linear')
pd_rain = pd.DataFrame({'date':rain.time,'rain_intensity':rain.values})
pd_rain['date'] = pd_rain.index
pd_rain = pd_rain.set_index('date')
pd_rain = pd_rain.resample('10T').interpolate('linear')

pd_rain_psd = pd.merge(pd_psd,pd_rain,on = 'date',how= 'outer')

#resample(time="1D").interpolate("linear")# 21145 item #[24:21097]
#xx  = rain>0.01
#xx = xx.astype(int) # 21145 item
pd_rain_psd['rain_flag'] = xx

pd_rain_psd_ccn = pd.merge(pd_rain_psd,wind_full,on = 'date',how = 'outer')
#%%
df_new = df # 21097
df_new['rain'] = rain
df_new['rain_flag'] = xx
#%%
trytry = psd.combine_first(rain)
#%%

