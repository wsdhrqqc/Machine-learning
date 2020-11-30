#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:26:22 2019

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

#%%Read in exhaust flag
    
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad02thresh']
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   
#%% Voy 1AB


 #% Handling ccn 
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
environ_path = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Aug = xr.open_mfdataset(glob.glob(environ_path), combine='nested',concat_dim='time')
dt_object = datetime.datetime.fromtimestamp(arm_ds_Aug['base_time'][3])

time_Aug = pd.to_datetime(arm_ds_Aug['seconds'][:], unit ='s', origin = pd.Timestamp(dt_object))  
df = pd.DataFrame({'lat':arm_ds_Aug['ship_lat'],'lon':arm_ds_Aug['ship_lon'],'sst':arm_ds_Aug['ship_irtsst']},index=time_Aug)

_, index1 = np.unique(df.index, return_index = True)
df  = df.iloc[index1]
df_1h = df.resample('1h').nearest()

environ_pathb = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1B*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Augb = xr.open_mfdataset(glob.glob(environ_pathb), combine='nested',concat_dim='time')

dt_objectb = datetime.datetime.fromtimestamp(arm_ds_Augb['base_time'][3])

time_Augb = pd.to_datetime(arm_ds_Augb['seconds'][:], unit ='s', origin = pd.Timestamp(dt_objectb))  
dfb = pd.DataFrame({'lat':arm_ds_Augb['ship_lat'],'lon':arm_ds_Augb['ship_lon'],'sst':arm_ds_Augb['ship_irtsst']},index=time_Augb)

_, index1 = np.unique(dfb.index, return_index = True)
dfb  = dfb.iloc[index1]
df_1hb = dfb.resample('1h').nearest()
df_full = pd.concat([df_1h,df_1hb])


wind_data = pd.read_csv('/Users/qingn/1029-1203true_wind.csv',parse_dates = True)
wind_data.rename(columns = {'Unnamed: 0':'time'}, inplace = True)
wind_data = wind_data.set_index('time')
wind_data.index=pd.to_datetime(wind_data.index)
#df_clip = df_full[5:]
df_wind_position = pd.concat([df_full, wind_data], axis=1,join = 'inner')


idx = find_closest(time_ccn.values, df_wind_position.index) 
ccn_con = con_full[idx] #'''This step is slow!!'''
qc_con = ccn_qc_full[idx]
ccn_con = np.ma.masked_where(qc_con!= 0, ccn_con)  #con after mask
df_wind_position['ccn'] = ccn_con

idx = find_closest(time_id_date, df_wind_position.index) 
flag_ccn = exhaust[idx] #'''This step is slow!!'''

index_ccn_mad = np.where(flag_ccn == 1)# pick up the contaminated time index
dirty = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn == 0)
clean = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)

df_wind_position2 = df_wind_position.iloc[clean]

first_bk_voyage =  pd.concat([df_wind_position2['2017-11-02':'2017-11-13 16'],df_wind_position2['2017-11-21 14':'2017-12-02 16']])
#third_bk_voyage = pd.concat([df_wind_position2['2018-01-16 01':'2018-01-26 22'],df_wind_position2['2018-02-20':'2018-03-03 22']])
#%% Figures
g = sns.jointplot(x = 'sst' , y = 'ccn', data = first_bk_voyage,kind = "kde", ylim=[0.01,300],color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'sst' , y = 'ccn', data = first_bk_voyage, ylim=[0.01,300],color="lightcoral")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 

g = sns.jointplot(x = 'lat' , y = 'ccn', data = first_bk_voyage,ylim=[0.01,300],kind = "kde",color="purple")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lat' , y = 'ccn', data = first_bk_voyage,ylim=[0.01,300],color="purple",kind="reg")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        
g = sns.jointplot(x = 'lon' , y = 'ccn', data = first_bk_voyage,ylim=[0.01,300],kind = "kde",color="blue")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lon' , y = 'ccn', data = first_bk_voyage,ylim=[0.01,300],color="blue",kind="reg")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   

g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = first_bk_voyage,ylim=[0.01,300],kind = "kde",color="green")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = first_bk_voyage,ylim=[0.01,300],color="green")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        
#%%

#VOY 3AB
path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201*.nc'
sdate = '20180116'
edate = '20180127'
files = glob.glob(path_ccn)
files.sort()
files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]
ccn_colavg= arm.read_netcdf(files)
ccn_colavg = ccn_colavg.resample(time = '1h').nearest()
con = ccn_colavg['N_CCN']
qc = ccn_colavg['qc_N_CCN']

sdate = '20180219'
edate = '20180304'
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
environ_path = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage3A*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Aug = xr.open_mfdataset(glob.glob(environ_path), combine='nested',concat_dim='time')
dt_object = datetime.datetime.fromtimestamp(arm_ds_Aug['base_time'][3])

time_Aug = pd.to_datetime(arm_ds_Aug['seconds'][:], unit ='s', origin = pd.Timestamp(dt_object))  
df = pd.DataFrame({'lat':arm_ds_Aug['ship_lat'],'lon':arm_ds_Aug['ship_lon'],'sst':arm_ds_Aug['ship_oisst']+273.15},index=time_Aug)

_, index1 = np.unique(df.index, return_index = True)
df  = df.iloc[index1]
df_1h = df.resample('1h').nearest()

environ_pathb = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage3B*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Augb = xr.open_mfdataset(glob.glob(environ_pathb), combine='nested',concat_dim='time')

dt_objectb = datetime.datetime.fromtimestamp(arm_ds_Augb['base_time'][3])

time_Augb = pd.to_datetime(arm_ds_Augb['seconds'][:], unit ='s', origin = pd.Timestamp(dt_objectb))  
dfb = pd.DataFrame({'lat':arm_ds_Augb['ship_lat'],'lon':arm_ds_Augb['ship_lon'],'sst':arm_ds_Augb['ship_irtsst']},index=time_Augb)

_, index1 = np.unique(dfb.index, return_index = True)
dfb  = dfb.iloc[index1]
df_1hb = dfb.resample('1h').nearest()
df_full = pd.concat([df_1h,df_1hb])


wind_data = pd.read_csv('/Users/qingn/0116-0304true_wind.csv',parse_dates = True)
wind_data.rename(columns = {'Unnamed: 0':'time'}, inplace = True)
wind_data = wind_data.set_index('time')
wind_data.index=pd.to_datetime(wind_data.index)
#df_clip = df_full[5:]
df_wind_position = pd.concat([df_full, wind_data], axis=1,join = 'inner')


idx = find_closest(time_ccn.values, df_wind_position.index) 
ccn_con = con_full[idx] #'''This step is slow!!'''
qc_con = ccn_qc_full[idx]
ccn_con = np.ma.masked_where(qc_con!= 0, ccn_con)  #con after mask
df_wind_position['ccn'] = ccn_con

idx = find_closest(time_id_date, df_wind_position.index) 
flag_ccn = exhaust[idx] #'''This step is slow!!'''

index_ccn_mad = np.where(flag_ccn == 1)# pick up the contaminated time index
dirty = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn == 0)
clean = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)

df_wind_position2 = df_wind_position.iloc[clean]

third_bk_voyage = pd.concat([df_wind_position2['2018-01-16 01':'2018-01-26 22'],df_wind_position2['2018-02-20':'2018-03-03 22']])
#%% Figures
g = sns.jointplot(x = 'sst' , y = 'ccn', data = third_bk_voyage,kind = "kde", ylim=[0.01,500],color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'sst' , y = 'ccn', data = third_bk_voyage, ylim=[0.01,500],color="lightcoral")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
#%%
g = sns.jointplot(x = 'lat' , y = 'ccn', data = third_bk_voyage,ylim=[0.01,500],kind = "kde",color="purple")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lat' , y = 'ccn', data = third_bk_voyage,ylim=[0.01,500],color="purple")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        #%%
g = sns.jointplot(x = 'lon' , y = 'ccn', data = third_bk_voyage,ylim=[0.01,500],kind = "kde",color="blue")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lon' , y = 'ccn', data = third_bk_voyage,ylim=[0.01,500],color="blue")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   

g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = third_bk_voyage,ylim=[0.01,500],kind = "kde",color="green")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = third_bk_voyage,ylim=[0.01,500],color="green")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
 

#%%
#VOY 2
path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201*.nc'
sdate = '20180105'
edate = '20180110'
files = glob.glob(path_ccn)
files.sort()
files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]
ccn_colavg= arm.read_netcdf(files)
ccn_colavg = ccn_colavg.resample(time = '1h').nearest()
con = ccn_colavg['N_CCN']
qc = ccn_colavg['qc_N_CCN']

time_ccn = con.time
#%%
environ_path = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage2*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Aug = xr.open_mfdataset(glob.glob(environ_path), combine='nested',concat_dim='time')
dt_object = datetime.datetime.fromtimestamp(arm_ds_Aug['base_time'][3])

time_Aug = pd.to_datetime(arm_ds_Aug['seconds'][:], unit ='s', origin = pd.Timestamp(dt_object))  
df = pd.DataFrame({'lat':arm_ds_Aug['ship_lat'],'lon':arm_ds_Aug['ship_lon'],'sst':arm_ds_Aug['ship_oisst']+273.15},index=time_Aug)

_, index1 = np.unique(df.index, return_index = True)
df  = df.iloc[index1]
df_1h = df.resample('1h').nearest()

wind_data = pd.read_csv('/Users/qingn/0105-0110true_wind.csv',parse_dates = True)
wind_data.rename(columns = {'Unnamed: 0':'time'}, inplace = True)
wind_data = wind_data.set_index('time')
wind_data.index=pd.to_datetime(wind_data.index)
#df_clip = df_full[5:]
df_wind_position = pd.concat([df_1h, wind_data], axis=1,join = 'inner')


idx = find_closest(time_ccn.values, df_wind_position.index) 
ccn_con = con_full[idx] #'''This step is slow!!'''
qc_con = ccn_qc_full[idx]
ccn_con = np.ma.masked_where(qc_con!= 0, ccn_con)  #con after mask
df_wind_position['ccn'] = ccn_con

idx = find_closest(time_id_date, df_wind_position.index) 
flag_ccn = exhaust[idx] #'''This step is slow!!'''

index_ccn_mad = np.where(flag_ccn == 1)# pick up the contaminated time index
dirty = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn == 0)
clean = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)

df_wind_position2 = df_wind_position.iloc[clean]

second_bk_voyage = df_wind_position2
#%% Figures
g = sns.jointplot(x = 'sst' , y = 'ccn', data = second_bk_voyage,kind = "kde", ylim=[0.01,500],color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'sst' , y = 'ccn', data = second_bk_voyage, ylim=[0.01,500],color="lightcoral")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
#%%
g = sns.jointplot(x = 'lat' , y = 'ccn', data = second_bk_voyage,ylim=[0.01,500],kind = "kde",color="purple")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18,) 
g = sns.jointplot(x = 'lat' , y = 'ccn', data = second_bk_voyage,ylim=[0.01,500],color="purple",kind="reg")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        #%%
g = sns.jointplot(x = 'lon' , y = 'ccn', data = second_bk_voyage,ylim=[0.01,500],kind = "kde",color="blue")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lon' , y = 'ccn', data = second_bk_voyage,ylim=[0.01,500],color="blue",kind="reg")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   

g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = second_bk_voyage,ylim=[0.01,500],kind = "kde",color="green")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = second_bk_voyage,ylim=[0.01,500],color="green",kind="reg")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 

#pd.concat([df_wind_position2['2018-01-16 01':'2018-01-26 22'],df_wind_position2['2018-02-20':'2018-03-03 22']])
#%% VOY4
path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201*.nc'
sdate = '20180309'
edate = '20180325'
files = glob.glob(path_ccn)
files.sort()
files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]
ccn_colavg= arm.read_netcdf(files)
ccn_colavg = ccn_colavg.resample(time = '1h').nearest()
con = ccn_colavg['N_CCN']
qc = ccn_colavg['qc_N_CCN']

time_ccn = con.time
#%%
environ_path = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage4*.cdf'
#environ_pathA = '/Users/qingn/Downloads/MARCUS Envrionmental prameters VAP v2/MARCUS_Environmental_Parameters_VAP_V2.0_Voyage1A*.cdf'
arm_ds_Aug = xr.open_mfdataset(glob.glob(environ_path), combine='nested',concat_dim='time')
dt_object = datetime.datetime.fromtimestamp(arm_ds_Aug['base_time'][3])

time_Aug = pd.to_datetime(arm_ds_Aug['seconds'][:], unit ='s', origin = pd.Timestamp(dt_object))  
df = pd.DataFrame({'lat':arm_ds_Aug['ship_lat'],'lon':arm_ds_Aug['ship_lon'],'sst':arm_ds_Aug['ship_oisst']+273.15},index=time_Aug)

_, index1 = np.unique(df.index, return_index = True)
df  = df.iloc[index1]
df_1h = df.resample('1h').nearest()

wind_data = pd.read_csv('/Users/qingn/0309-0324true_wind.csv',parse_dates = True)
wind_data.rename(columns = {'Unnamed: 0':'time'}, inplace = True)
wind_data = wind_data.set_index('time')
wind_data.index=pd.to_datetime(wind_data.index)
#df_clip = df_full[5:]
df_wind_position = pd.concat([df_1h, wind_data], axis=1,join = 'inner')


idx = find_closest(time_ccn.values, df_wind_position.index) 
ccn_con = con_full[idx] #'''This step is slow!!'''
qc_con = ccn_qc_full[idx]
ccn_con = np.ma.masked_where(qc_con!= 0, ccn_con)  #con after mask
df_wind_position['ccn'] = ccn_con

idx = find_closest(time_id_date, df_wind_position.index) 
flag_ccn = exhaust[idx] #'''This step is slow!!'''

index_ccn_mad = np.where(flag_ccn == 1)# pick up the contaminated time index
dirty = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn == 0)
clean = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)

df_wind_position2 = df_wind_position.iloc[clean]
forth_bk_voyage = df_wind_position2[:'2018-03-11 21']

#%% Figures
g = sns.jointplot(x = 'sst' , y = 'ccn', data = forth_bk_voyage,kind = "kde", ylim=[0.01,500],color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'sst' , y = 'ccn', data = forth_bk_voyage, ylim=[0.01,500],color="lightcoral",kind="reg")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
#%%
g = sns.jointplot(x = 'lat' , y = 'ccn', data = forth_bk_voyage,ylim=[0.01,500],kind = "kde",color="purple")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lat' , y = 'ccn', data = forth_bk_voyage,ylim=[0.01,500],color="purple",kind="reg")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        #%%
g = sns.jointplot(x = 'lon' , y = 'ccn', data = forth_bk_voyage,ylim=[0.01,500],kind = "kde",color="blue")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lon' , y = 'ccn', data = forth_bk_voyage,ylim=[0.01,500],color="blue",kind="reg")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   

g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = forth_bk_voyage,ylim=[0.01,500],kind = "kde",color="green")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = forth_bk_voyage,ylim=[0.01,500],color="green",kind="reg")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
#%% Combine 4 voyage

Whole_voyage = pd.concat([first_bk_voyage,second_bk_voyage,third_bk_voyage,forth_bk_voyage])
cpc_wind_fullname = '/Users/qingn/four_voyage_env_ccn.csv'#%indate
Whole_voyage.to_csv(cpc_wind_fullname)
#%% Figures
g = sns.jointplot(x = 'sst' , y = 'ccn', data = Whole_voyage,kind = "kde", ylim=[0.01,500],color="lightcoral")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'sst' , y = 'ccn', data = Whole_voyage, ylim=[0.01,500],color="lightcoral",kind="reg")
#sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")
g.set_axis_labels('surface temperature(K)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
#%%
g = sns.jointplot(x = 'lat' , y = 'ccn', data = Whole_voyage,ylim=[0.01,500],kind = "kde",color="purple")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lat' , y = 'ccn', data = Whole_voyage,ylim=[0.01,500],color="purple",kind="reg",kind="reg")
g.set_axis_labels('latitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        #%%
g = sns.jointplot(x = 'lon' , y = 'ccn', data = Whole_voyage,ylim=[0.01,500],kind = "kde",color="blue")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lon' , y = 'ccn', data = Whole_voyage,ylim=[0.01,500],color="blue",kind="reg",kind="reg")
g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   

g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = Whole_voyage,ylim=[0.01,500],kind = "kde",color="g")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'true_speed' , y = 'ccn', data = Whole_voyage,ylim=[0.01,500],color="greens",kind="reg")
g.set_axis_labels('wind_speed(m/s)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 

g = sns.jointplot(x = 'lat' , y ='sst' , data = Whole_voyage,kind = "kde",color="blue")
#g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18) 
g = sns.jointplot(x = 'lat' , y = 'sst', data = Whole_voyage,color="blue",kind="reg")
#g.set_axis_labels('longitude(degree)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 18)   
        
        