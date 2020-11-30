#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 01:51:28 2019
produce wd ws per h
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
    return file
#%%
# read in met and met data to calculate natural wind speed and wind direction
def correct_nature_wind(objectname_met, objectname_nav):
    '''Read in the nav and met object and use the algorithm from
    NOAA Technical Memorandum OAR (A GUIDE TO MAKING CLIMATE QUALITY METEOROLOGICAL AND FLUX MEASUREMENTS AT SEA)
    https://www.go-ship.org/Manual/fluxhandbook_NOAA-TECH%20PSD-311v3.pdf 
    to tranfer the ship-relative winds into the true/natural-relative winds
    for both direction(dirt) and wind speed(ut)
    : param objectname_met: the met data
    : param objectname_nav: the nav data 
    : returns: A list of <xarray.Dataset>'''
 
    wspd_name='wind_speed';wdir_name='wind_direction'
    heading_name='yaw';cog_name='course_over_ground'
    sog_name='speed_over_ground'
# Set variables to be used and convert to radians
    rels = objectname_met[wspd_name]
#    unit_ut = met[wspd_name].units
    reld = np.deg2rad(objectname_met[wdir_name])
#    reld_deg = met[wdir_name]
    
    head = np.deg2rad(objectname_nav[heading_name])
#    head_deg = nav[heading_name]
    cog = np.deg2rad(objectname_nav[cog_name])
    sog = objectname_nav[sog_name]
    # Calculate winds based on method in the document denoted above
    relsn = rels * np.cos(head + reld)
    relse = rels * np.sin(head + reld)
    
    sogn = sog * np.cos(cog)
    soge = sog * np.sin(cog)
    
    un = relsn - sogn
    ue = relse - soge
    
    dirt = np.mod(np.rad2deg(np.arctan2(ue, un)) + 360., 360)
    ut = np.sqrt(un ** 2. + ue ** 2.)
    return dirt, ut, sog

#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

met_dir = '/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201*.nc'#%indate
#/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.20171025.000000.nc
nav_dir = '/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.201*'#%indate
#%%
sdate = '20171029'
edate = '20180324'
met = arm_read_netcdf(met_dir,sdate,edate)

reld = met['wind_direction']
rels = met['wind_speed']

date = [201711,201712,201801,201802,201803]
wind_whole = pd.DataFrame()
for indate in date:
    wind_data11 = pd.read_csv('/Users/qingn/%scpc_wind_lat_.csv'%indate,parse_dates = True)
    wind_data11.rename(columns = {'Unnamed: 0':'time'}, inplace = True)
    
    wind_data11 = wind_data11.set_index('time')
    wind_data11.index=pd.to_datetime(wind_data11.index)
    
    wind_data11 = wind_data11.resample('1h').nearest()
    print(np.shape(wind_data11))
    wind_whole=wind_whole.append(wind_data11)
#    wind_whole = pd.concat([wind_whole,wind_data11],axis=1,join='inner')
trues= wind_whole['wind_speed']
trued = wind_whole['wind_direction']

#%%
#plot figures
kwargs = dict(alpha=0.5, bins=25, density=True, stacked=True)


fig = plt.figure(figsize = (FIGWIDTH,FIGHEIGHT))
plt.hist(trued, **kwargs,label = 'true wind direction')
#
#plt.title('histogram for true wind direction')
#plt.xlabel('degree')
#plt.ylabel('counts')

plt.hist(reld, **kwargs,label='relative wind direction')
plt.gca().set(title='Probability Histogram of True Wind Direction', ylabel='Probability',xlabel='degree')
#plt.hist(x3, **kwargs, color='r', label='Good')
#plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')

plt.legend(prop={'size': 20})
fdir = '/Users/qingn/Desktop/NQ/'
#fig.tight_layout()
#plt.show()
#plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.savefig(fdir+'true_relative_wind_direction.png',bbox_inches = 'tight')  


#%% Normalize histogram plot for wind speed
plt.figure(figsize = (FIGWIDTH,FIGHEIGHT))
plt.hist(trues, **kwargs, color = 'green', label = 'true wind speed')
plt.hist(rels, **kwargs, color='blue',label='relative wind speed')

plt.gca().set(title='Probability Histogram of True Wind Speed', ylabel='Probability',xlabel='m/s')

#plt.legend()
plt.legend(prop={'size': 16})
#plt.savefig()

plt.savefig(fdir+'true_relative_wind_speed.png',bbox_inches = 'tight')  

#plt.legend(markerscale=3,ncol=4, bbox_to_anchor=(0.5, -0.2),
#                  loc=10, fontsize='small')
#axs[p].set_ylim(ylim)


 #%%
sdate = '20171120'
edate = '20171203'

met1 = arm_read_netcdf(met_dir,sdate,edate)
nav1 = arm_read_netcdf(nav_dir,sdate,edate)
dirt1, ut1, sog1 = correct_nature_wind(met1, nav1)

df = pd.DataFrame({'true_direction':dirt,'true_speed':ut,'speed_over_ground':sog},index=ut.time.values)
df1 = pd.DataFrame({'true_direction':dirt1,'true_speed':ut1,'speed_over_ground':sog1},index=ut1.time.values)
wind_fullname = '/Users/qingn/1029-1203true_wind.csv'#%indate
df_full = pd.concat([df,df1])
df_full.to_csv(wind_fullname)

#%%
sdate = '20180105'
edate = '20180110'

met1 = arm_read_netcdf(met_dir,sdate,edate)
nav1 = arm_read_netcdf(nav_dir,sdate,edate)
dirt1, ut1, sog1 = correct_nature_wind(met1, nav1)


df1 = pd.DataFrame({'true_direction':dirt1,'true_speed':ut1,'speed_over_ground':sog1},index=ut1.time.values)
wind_fullname = '/Users/qingn/0105-0110true_wind.csv'#%indate
#df_full = pd.concat([df,df1])
df1.to_csv(wind_fullname)
#%%

sdate = '20180309'
edate = '20180324'

met1 = arm_read_netcdf(met_dir,sdate,edate)
nav1 = arm_read_netcdf(nav_dir,sdate,edate)
dirt1, ut1, sog1 = correct_nature_wind(met1, nav1)


df1 = pd.DataFrame({'true_direction':dirt1,'true_speed':ut1,'speed_over_ground':sog1},index=ut1.time.values)
wind_fullname = '/Users/qingn/0309-0324true_wind.csv'#%indate

df1.to_csv(wind_fullname)
#%%

sdate = '20180116'
edate = '20180127'

met = arm_read_netcdf(met_dir,sdate,edate)
nav = arm_read_netcdf(nav_dir,sdate,edate)

dirt, ut, sog = correct_nature_wind(met, nav)

sdate = '20180219'
edate = '20180304'

met1 = arm_read_netcdf(met_dir,sdate,edate)
nav1 = arm_read_netcdf(nav_dir,sdate,edate)
dirt1, ut1, sog1 = correct_nature_wind(met1, nav1)

df = pd.DataFrame({'true_direction':dirt,'true_speed':ut,'speed_over_ground':sog},index=ut.time.values)
df1 = pd.DataFrame({'true_direction':dirt1,'true_speed':ut1,'speed_over_ground':sog1},index=ut1.time.values)
wind_fullname = '/Users/qingn/0116-0304true_wind.csv'#%indate
df_full = pd.concat([df,df1])
df_full.to_csv(wind_fullname)

# =============================================================================
# 
# 
# 
# #%% implement flag on cpc concentration(2)
#     idx = find_closest(time_id_date, time_cpc) 
#     flag_cpc = exhaust[idx] #'''This step is slow!!'''
#     #%%
#     index_cpc_mad = np.where(flag_cpc == 1)# pick up the contaminated time index
#     dirty = np.array(index_cpc_mad[0]) # name the index to be dirty(contaminated index)
#     index_cpc_clean_mad = np.where(flag_cpc == 0)
#     clean = np.array(index_cpc_clean_mad[0])# name the index to be clean(clean index)
# 
# #%%
#     cpc_con1 = np.ma.masked_where(qc_cpc!= 0, cpc_con)
#     #win_cpc_df = pd.DataFrame({'wind_speed':ut, 'wind_direction':dirt, 'cpc_con':cpc_con1}, index = time_cpc)
#     # to make a DataFrame with the cleaned data:
#     #win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean], 'wind_direction':dirt[clean], 'cpc_con':cpc_con1[clean]}, index = time_cpc[clean])
#     #%% We aim to save the file in case always need to re-run the data and program
#     # True to always run cross validation, false to re-load existing run
#     # or run cross validation for the first time
#     cpc_wind_fullname = '/Users/qingn/%scpc_wind_lat_.csv'%indate
#     win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean], 'wind_direction':dirt[clean], 'cpc_con':cpc_con1[clean],'lat':nav['lat'][clean]}, index = time_cpc[clean])
#     #joblib.dump(win_cpc_df1, cpc_wind_fullname)
#     win_cpc_df1.to_csv(cpc_wind_fullname)
# =============================================================================
