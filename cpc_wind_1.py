#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:38:58 2019

@author: qingn
@Unpublish data, please keep it confidential
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:27:23 2019

@author: qingn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:32:16 2019
THis should be done with a few function, I personally begin doing all of this
in a long script and then invide them into differnt part by setting up functions
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
FONTSIZE = 18
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['axes.labelsize']=FONTSIZE
plt.rcParams['axes.titlesize']=FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE) 

HOME_DIR = str(pathlib.Path.home()/'Desktop/NQ' )
#%%

#pathlib.Path.cwd()

#%%

def arm_read_netcdf(directory_filebase):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    
    file_dir = str(HOME_DIR + directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time='10s').nearest()
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
    rels = met[wspd_name]
#    unit_ut = met[wspd_name].units
    reld = np.deg2rad(met[wdir_name])
#    reld_deg = met[wdir_name]
    
    head = np.deg2rad(nav[heading_name])
#    head_deg = nav[heading_name]
    cog = np.deg2rad(nav[cog_name])
    sog = nav[sog_name]
    # Calculate winds based on method in the document denoted above
    relsn = rels * np.cos(head + reld)
    relse = rels * np.sin(head + reld)
    
    sogn = sog * np.cos(cog)
    soge = sog * np.sin(cog)
    
    un = relsn - sogn
    ue = relse - soge
    
    dirt = np.mod(np.rad2deg(np.arctan2(ue, un)) + 360., 360)
    ut = np.sqrt(un ** 2. + ue ** 2.)
    return dirt, ut

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
#indate = 201711

# EXHAUST_ID
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad01thresh']
time_id = np.array(exhaust_id['time'])
# From the number[0,1,2,3,4...] into the time stamps
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   
#%%
indate_index = [201712,201801,201802,201803]#[201711]

for indate in indate_index:
    cpc_dir = '/maraoscpc/maraoscpcfM1.b1.%s*'%indate
    met_dir = '/maraosmet/maraosmetM1.a1/maraosmetM1.a1.%s*.nc'%indate
    nav_dir = '/marnav/marnavbeM1.s1.%s*'%indate
    cpc = arm_read_netcdf(cpc_dir)
    met = arm_read_netcdf(met_dir)
    nav = arm_read_netcdf(nav_dir)
    
    cpc_con = cpc['concentration']
    qc_cpc = cpc['qc_concentration']
    time_cpc = cpc['time'].values
    cpc.close()
#%%
    dirt, ut = correct_nature_wind(met, nav)
    met.close()
#nav.close()

#%% implement flag on cpc concentration(2)
    idx = find_closest(time_id_date, time_cpc) 
    flag_cpc = exhaust[idx] #'''This step is slow!!'''
    #%%
    index_cpc_mad = np.where(flag_cpc == 1)# pick up the contaminated time index
    dirty = np.array(index_cpc_mad[0]) # name the index to be dirty(contaminated index)
    index_cpc_clean_mad = np.where(flag_cpc == 0)
    clean = np.array(index_cpc_clean_mad[0])# name the index to be clean(clean index)

#%%
    cpc_con1 = np.ma.masked_where(qc_cpc!= 0, cpc_con)
    #win_cpc_df = pd.DataFrame({'wind_speed':ut, 'wind_direction':dirt, 'cpc_con':cpc_con1}, index = time_cpc)
    # to make a DataFrame with the cleaned data:
    #win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean], 'wind_direction':dirt[clean], 'cpc_con':cpc_con1[clean]}, index = time_cpc[clean])
    #%% We aim to save the file in case always need to re-run the data and program
    # True to always run cross validation, false to re-load existing run
    # or run cross validation for the first time
    cpc_wind_fullname = '/Users/qingn/%scpc_wind_lat_.csv'%indate
    win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean], 'wind_direction':dirt[clean], 'cpc_con':cpc_con1[clean],'lat':nav['lat'][clean]}, index = time_cpc[clean])
    #joblib.dump(win_cpc_df1, cpc_wind_fullname)
    win_cpc_df1.to_csv(cpc_wind_fullname)
#    reset
##%%
#
#
#force = False 
#
#
#if force or (not os.path.exists(cpc_wind_fullname)):
#    win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean], 'wind_direction':dirt[clean], 'cpc_con':cpc_con1[clean]}, index = time_cpc[clean])
#    joblib.dump(win_cpc_df1, cpc_wind_fullname)
#else:
#    # Re-load saved crossval object instead of re-running
#    win_cpc_df1 = joblib.load(cpc_wind_fullname)
## %%
##% Plotting
#from scipy.stats import gaussian_kde
#x = win_cpc_df1['cpc_con']
#y = win_cpc_df1['wind_speed']
#z = win_cpc_df1['lat']
#fig, ax = plt.subplots()
##plt.scatter(win_cpc_df['cpc_con'][clean2],win_cpc_df['wind_speed'][clean2])
## a lot of overplotting
##ax.scatter(x, y, c=z, s=100, edgecolor='')
##ax.hexbin(x, y, gridsize=20, cmap=plt.cm.BuGn_r)
##im = ax.hexbin(y, x, gridsize=20, bins = 'log', 
##               cmap=plt.cm.BuGn)
#im = ax.hexbin(x, y, gridsize=25, cmap=plt.cm.BuGn)
##ax.set_xlim(0,2000)
## bins ='log' 
##plt.xlabel('1/cc')
#ax.set_xlabel('1/cc')
##axs[p].set_ylim((0,25))
#ax.set_ylabel('m/s')
#ax.set_title('wind speed & CPC concentration')
##ax.colorbar()
#cb = fig.colorbar(im, ax=ax)
#cb.set_label('counts')
##cb.set_label('log10(N)')
#plt.show()
#
##%
#sns.set()
#g = sns.jointplot(y = 'wind_speed',x='cpc_con' ,data = win_cpc_df1, color="purple")
#g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)
#
#g = sns.jointplot(x = 'cpc_con' , y = 'wind_direction', data = win_cpc_df1,color="lightcoral")
#g.set_axis_labels('cpc concentration(1/cc)','wind direction(degree)', fontweight = 'bold',fontsize = 18)
#
#g = sns.jointplot(x = 'cpc_con' , y = 'lat', data = win_cpc_df1,color="blue")
#g.set_axis_labels('cpc concentration(1/cc)','latitude(south_degree)', fontweight = 'bold',fontsize = 18)
#
## %
## Jointplot - Scatterplot and Histogram
#sns.set()
#g = sns.jointplot(x = 'cpc_con' , y = 'wind_speed', data = win_cpc_df1, kind ="hex", color="lightcoral")
##,xlim=[1,1800],ylim=[0.01,25]
#g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)
#g = sns.jointplot(x = 'cpc_con' , y = 'wind_direction', data = win_cpc_df1, kind ="hex",color="lightcoral")
#g.set_axis_labels('cpc concentration(1/cc)','degree relative to true north', fontweight = 'bold',fontsize = 18)
#
#g = sns.jointplot(x = 'cpc_con' , y = 'lat', data = win_cpc_df1, kind ="hex",color="lightcoral")
#g.set_axis_labels('cpc concentration(1/cc)','latitude(south)', fontweight = 'bold',fontsize = 18)
#
##,xlim=[1,1800]
##data,xlim=[1,100],ylim=[0.01,100]
##ax.set_xscale('log')
##ax.set_yscale('log')
##%
#
##g.ax_marg_x.set_xscale('log')
##g.ax_marg_y.set_yscale('log')
##g.ax_joint.set_xscale('log')
##g.ax_joint.set_yscale('log')
##%
## Jointplot - Scatterplot and Histogram
#sns.set()
#g=sns.jointplot(x = 'cpc_con', y = 'wind_speed', data = win_cpc_df1,kind = "kde", color="purple") # contour plot
#g.set_axis_labels('cpc concentration(1/cc)','wind speed(m/s)', fontweight = 'bold',fontsize = 18)
#g=sns.jointplot(x = 'cpc_con', y = 'wind_direction', data = win_cpc_df1,kind = "kde", color="purple") # contour plot
#g.set_axis_labels('cpc concentration(1/cc)','degree relative to true north', fontweight = 'bold',fontsize = 18)
#
#g=sns.jointplot(x = 'cpc_con', y = 'lat', data = win_cpc_df1,kind = "kde", color="purple") # contour plot
#g.set_axis_labels('cpc concentration(1/cc)','south latitude', fontweight = 'bold',fontsize = 18)
#
###%% This block does exactly the same thing that the function arm_read_netcdf has done
### read in cpc,met,nav data
##
##cpc_dir = str(HOME_DIR + '/maraoscpc/maraoscpcfM1.b1.201801*')
##cpc_ori = arm.read_netcdf(cpc_dir)
##_, index1 = np.unique(cpc_ori['time'], return_index = True)
##cpc_ori = cpc_ori.isel(time = index1)
##cpc = cpc_ori.resample(time='10s').nearest()
##
##met_dir = str(HOME_DIR + '/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201801*.nc')
##met_ori = arm.read_netcdf(met_dir)
##_, index1 = np.unique(met_ori['time'], return_index = True)
##met_ori = met_ori.isel(time = index1)
##met = met_ori.resample(time='10s').nearest()
##
##nav_dir = str(HOME_DIR + '/marnav/marnavbeM1.s1.201801*')
##nav_ori = arm.read_netcdf(nav_dir)
##_, index1 = np.unique(nav_ori['time'], return_index = True)
##nav_ori = nav_ori.isel(time = index1)
##nav = nav_ori.resample(time='10s').nearest()
##
###nav = nav_ori.resample(time='1s').nearest()
###met = met_ori
##cpc_con = cpc['concentration']
##qc_cpc = cpc['qc_concentration']
##time_cpc = cpc['time'].values
##
###%% The following block did exactly the same thing as function correct_nature_wind
##wspd_name='wind_speed';wdir_name='wind_direction'
##heading_name='yaw';cog_name='course_over_ground'
##sog_name='speed_over_ground'
### Set variables to be used and convert to radians
##rels = met[wspd_name]
##unit_ut = met[wspd_name].units
##reld = np.deg2rad(met[wdir_name])
##reld_deg = met[wdir_name]
##
##head = np.deg2rad(nav[heading_name])
##head_deg = nav[heading_name]
##cog = np.deg2rad(nav[cog_name])
##sog = nav[sog_name]
### Calculate winds based on method in the document denoted above
##relsn = rels * np.cos(head + reld)
##relse = rels * np.sin(head + reld)
##
##sogn = sog * np.cos(cog)
##soge = sog * np.sin(cog)
##
##un = relsn - sogn
##ue = relse - soge
##
##dirt = np.mod(np.rad2deg(np.arctan2(ue, un)) + 360., 360)
##ut = np.sqrt(un ** 2. + ue ** 2)