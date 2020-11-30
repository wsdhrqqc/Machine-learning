#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:35:36 2019
corrected CCN, we make the datestream work first and then correct the data to make it beautiful
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

def arm_read_netcdf(directory_filebase, time_resolution):
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
    file = file_ori.resample(time = time_resolution).nearest()
    return file

#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
#%%
    
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad02thresh']
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   

path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.20171*.nc'

sdate = '20171029'
edate = '20171114'
files = glob.glob(path_ccn)
files.sort()
files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]
ccn_colavg= arm.read_netcdf(files)
ccn_colavg = ccn_colavg.resample(time = '1h').nearest()
con = ccn_colavg['N_CCN']
qc = ccn_colavg['qc_N_CCN']

sdate = '20171120'
edate = '20171203'
files1 = glob.glob(path_ccn)
files1.sort()
files1 = [f for f in files1 if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]
#file = files+files1

ccn_colavg1= arm.read_netcdf(files1)
ccn_colavg1 = ccn_colavg1.resample(time = '1h').nearest()
con1 = ccn_colavg1['N_CCN']
qc1 = ccn_colavg1['qc_N_CCN']
#con_full = xr.concat([con,con1],dim =index )
con_full = xr.combine_nested([con,con1],concat_dim=['time'])[:,3] # supersaturation_setpoint  float32 0.5
ccn_qc_full = xr.combine_nested([qc,qc1],concat_dim=['time'])[:,3] # supersaturation_setpoint  float32 0.5
time_ccn = con_full.time


#%%
#qing_exhau = netCDF4.Dataset('/Users/qingn/20171030201180324qing_flag_00.cdf')
#qing_exhaust = qing_exhau['exhaust_flag']
#environ_path = '/Users/qingn/Downloads/drive-download-20191125T073459Z-001/MARCUS_V1_VAP_20171029.cdf'
# =============================================================================
# environ_path1 = '/Users/qingn/Downloads/MARCUS VAP/MARCUS_*.cdf'\
# 
# files_env_July = glob.glob(environ_path1)
# files_env_July.sort()
# #arm_ds = xr.open_mfdataset(files_env_July, combine='nested', concat_dim)
# arm_ds_July = xr.open_mfdataset(files_env_July, combine='nested',concat_dim='time')
# arm_July = arm_ds_July.load()
# timestamp = datetime.datetime.timestamp(datetime.datetime(2017, 10, 28, 18, 0))
# base_time_July = arm_July['base_time']
# base_time_July[:3170] = timestamp
# 
# time_July = arm_ds_July['seconds'].load()
# 
# time_July[:3170] = timestamp + time_July[:3170]
# df = pd.DataFrame({'precipitation':arm_ds_July['precipitation_flag_sfc'], 'lts':arm_ds_July['lts'],'lat':arm_ds_July['ship_lat'],'lon':arm_ds_July['ship_lon']},index=time_env)
# 
# =============================================================================

environ_path='/Users/qingn/Downloads/MARCUS VAP/MARCUS_*29*.cdf'
#environ_path = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Aug = xr.open_mfdataset(glob.glob(environ_path), combine='nested',concat_dim='time')

pars2 = arm_ds_Aug['precipitation_flag_sfc']
dt_object = datetime.datetime.fromtimestamp(arm_ds_Aug['base_time'][3])

time_Aug = pd.to_datetime(arm_ds_Aug['seconds'][:], unit ='s', origin = pd.Timestamp(dt_object))  
df = pd.DataFrame({'lat':arm_ds_Aug['ship_lat'],'lon':arm_ds_Aug['ship_lon'],'pre':arm_ds_Aug['precipitation_flag_sfc']},index=time_Aug)

_, index1 = np.unique(df.index, return_index = True)
df  = df.iloc[index1]
df_1h = df.resample('1h').nearest()
#df_1h = df_1h[29:]
#%%
# =============================================================================
# 
# environ_pathb = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1B*.cdf'
# #environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
# arm_ds_Augb = xr.open_mfdataset(glob.glob(environ_pathb), combine='nested',concat_dim='time')
# 
# #time_Aug = 
# dt_objectb = datetime.datetime.fromtimestamp(arm_ds_Augb['base_time'][3])
# 
# time_Augb = pd.to_datetime(arm_ds_Augb['seconds'][:], unit ='s', origin = pd.Timestamp(dt_objectb))  
# dfb = pd.DataFrame({'lat':arm_ds_Augb['ship_lat'],'lon':arm_ds_Augb['ship_lon'],'sst':arm_ds_Augb['ship_irtsst']},index=time_Augb)
# 
# _, index1 = np.unique(dfb.index, return_index = True)
# dfb  = dfb.iloc[index1]
# df_1hb = dfb.resample('1h').nearest()
# df_full = pd.concat([df_1h,df_1hb])

# =============================================================================


wind_data = pd.read_csv('/Users/qingn/1029-1203true_wind.csv',parse_dates = True)

wind_data.rename(columns = {'Unnamed: 0':'time'}, inplace = True)
wind_data = wind_data.set_index('time')
wind_data.index=pd.to_datetime(wind_data.index)
#df_clip = df_full[5:]
df_wind_position = pd.concat([df_1h, wind_data], axis=1,join = 'inner')


idx = find_closest(time_ccn.values, df_wind_position.index) 
ccn_con = con_full[idx] #'''This step is slow!!'''
qc_con = ccn_qc_full[idx]
ccn_con = np.ma.masked_where(qc_con!= 0, ccn_con)
df_wind_position['ccn'] = ccn_con

idx = find_closest(time_id_date, df_1h.index) 
flag_ccn = exhaust[idx] #'''This step is slow!!'''
#_, index1 = np.unique(ccn_colavg['time'], return_index = True)
#ccn_colavg = ccn_colavg.isel(time = index1)
#con = ccn_colavg['N_CCN']

##con = con[np.where(qc_con==0)[0]]
#time_ccn = ccn_colavg['time'].values
#%


#%%
index_ccn_mad = np.where(flag_ccn == 1)# pick up the contaminated time index
dirty = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn == 0)
clean = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)
 #con after mask
df_wind_position2 = df_wind_position.iloc[clean]
#df_wind_position2 = pd.DataFrame({'ccn_con':ccn_con1[clean],'lat':d[clean],'lon':e[clean],'sst':f[clean]},index=df_1h.index[clean])
#matrix_after_transpose = np.transpose(df_wind_position2[[ 'lon','lat','speed_over_ground']].values)

#%%sns.set(),ylim=[0.01,300]
g = sns.jointplot(x = 'sst' , y = 'ccn', data = df_wind_position2,kind = "kde", ylim=[0.01,300],color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'sst' , y = 'ccn', data = df_wind_position2, ylim=[0.01,300],color="lightcoral")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
#%%
g = sns.jointplot(x = 'sst' , y = 'lat', data = df_wind_position2,kind = "kde",color="lightcoral")
g = sns.jointplot(x = 'sst' , y = 'lat', data = df_wind_position2,color="lightcoral")
g.set_axis_labels('surface temperature(K)','latitude',fontweight = 'bold',fontsize = 18) 
#%%
g = sns.jointplot(x = 'lat' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],kind = "kde",color="lightcoral")
g = sns.jointplot(x = 'lat' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],color="lightcoral")
#
#%%
g = sns.jointplot(x = 'lon' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],kind = "kde",color="lightcoral")
g = sns.jointplot(x = 'lon' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],color="lightcoral")
#%%
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],kind = "kde",color="lightcoral")
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],color="lightcoral")
#%%
g = sns.jointplot(x = 'true_direction' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],kind = "kde",color="lightcoral")
g = sns.jointplot(x = 'true_direction' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],color="lightcoral")
#%%

g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],kind = "kde",color="lightcoral")
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = df_wind_position2,ylim=[0.01,300],color="lightcoral")


 
environ_path_b = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1B_20171121_20171203.cdf'
env1 = netCDF4.Dataset(environ_path1)
env = netCDF4.Dataset(environ_path)
#pars2 = env['rain_flag_pars2']
a = env['base_time'][:]
a1 = env1['base_time'][:]
dt_object = datetime.datetime.fromtimestamp(a)
dt_object1 = datetime.datetime.fromtimestamp(a1)
#time = dt_object +env['seconds']
time_env = pd.to_datetime(env['seconds'][:], unit ='s', origin = pd.Timestamp(dt_object))  
#env_set = xr.DataArray([env['rain_flag_pars2'],env['lts']], coords=[time_env], dims=['time'])

#_, index1 = np.unique(env_set['time'], return_index = True)
#env_set = env_set.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
#pars2_ori = env_set.resample(time='1h').nearest()
#pars2 = pars2_ori[29:]
#df = pd.DataFrame({'precipitation':env['rain_flag_pars2'], 'lts':env['lts'],'lat':env['ship_lat'],'lon':env['ship_lon']},index=time_env)
df_v2 = pd.DataFrame({'lat':env['ship_lat'],'lon':env['ship_lon'],'sst':env['ship_irtsst']},index=time_env)

_, index1 = np.unique(df.index, return_index = True)
df  = df.iloc[index1]
df_1h = df.resample('1h').nearest()
df_1h = df_1h[29:]
c = np.ma.masked_where(df_1h['lts']== -9999, df_1h['lts'])
d = np.ma.masked_where(df_1h['lat']== -9999, df_1h['lat'])
e = np.ma.masked_where(df_1h['lon']== -9999, df_1h['lon'])
f = np.ma.masked_where(df_v2['sst']== -9999, df_v2['sst'])
#df_clean = pd.DataFrame({'precipitation':df_1h['precipitation'][clean], 'lts':c[clean],'ccn_con':ccn_con1[clean],'lat':d[clean],'lon':e[clean]},index=df_1h.index[clean])
df_v2_clean = pd.DataFrame({'ccn_con':ccn_con1[clean],'lat':d[clean],'lon':e[clean],'sst':f[clean]},index=df_1h.index[clean])
#%%

win_cpc_df1 = pd.DataFrame({'precipitation':pars2[clean], 'cpc_con':ccn_con1[clean]}, index = time_ccn[clean])
win_cpc_df = win_cpc_df1.dropna()
#%%

rain = np.where(win_cpc_df['precipitation']==1)[0]
nothing = np.where(win_cpc_df['precipitation']==0)[0]
boxplot = plt.boxplot([win_cpc_df['cpc_con'][rain], win_cpc_df['cpc_con'][nothing]], patch_artist=True, sym='.')
boxplot['boxes'][0].set_facecolor('pink')
boxplot['boxes'][1].set_facecolor('lightblue')
plt.xticks(ticks=[1, 2], labels=['rain', 'clear'])
plt.xlabel("Class")
plt.title("CCN in rain and clear weather(ss=0.5%)")
#%%
#data = [data, d2, d2[::2,0]]
fig7, ax7 = plt.subplots()
ax7.set_title('Multiple Samples with Different sizes')
rain = np.where(win_cpc_df1['precipitation']==1)[0]
nothing = np.where(win_cpc_df1['precipitation']==0)[0]
ax7.boxplot(win_cpc_df['cpc_con'][rain],win_cpc_df['cpc_con'][nothing])
#ax7.boxplot(win_cpc_df1['cpc_con'][nothing])
plt.show()

#%%
sns.set()
g = sns.jointplot(x = 'lts' , y = 'ccn_con', data = df_clean,kind = "kde",ylim=[0.01,300], color="lightcoral")
g.set_axis_labels('Lower tropospheric stability(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)
g = sns.jointplot(x = 'lts' , y = 'ccn_con', data = df_clean,ylim=[0.01,300], color="lightcoral")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('Lower tropospheric stability(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)
 #%%
g = sns.jointplot(x = 'lat' , y = 'ccn_con', data = df_clean,kind = "kde",ylim=[0.01,300], color="lightcoral")
g.set_axis_labels('latitude(S)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)
g = sns.jointplot(x = 'lon' , y = 'ccn_con', data = df_clean,kind = "kde",ylim=[0.01,300], color="lightcoral")
g = sns.jointplot(x = 'lon' , y = 'ccn_con', data = df_clean,ylim=[0.01,300], color="lightcoral")
g.set_axis_labels('longitude(E)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)        
#%%

g = sns.jointplot(x = 'sst' , y = 'ccn_con', data = df_v2_clean, ylim=[0.01,300], color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
        #%%
g = sns.jointplot(x = 'sst' , y = 'ccn_con', data = df_v2_clean,kind = "kde",xlim = [280,290],ylim=[0.01,300], color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
        
        #%%
plt.figure(figsize = (12,6))

plt.plot(time_ccn,con[:,4])