#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:27:23 2019
@Unpublish data, please keep it confidential
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
#%%
# readin cpc data
cpc_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201801*')
_, index1 = np.unique(cpc_ori['time'], return_index = True)
cpc_ori = cpc_ori.isel(time = index1)
cpc = cpc_ori.resample(time='10s').nearest()
cpc_con = cpc['concentration']
qc_cpc = cpc['qc_concentration']
time_cpc = cpc['time'].values

# EXHAUST_ID
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad01thresh']
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   

#%%
# read in met and met data to calculate natural wind speed and wind direction

met_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201801*.nc')
nav_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.201801*')
_, index1 = np.unique(met_ori['time'], return_index = True)
met_ori = met_ori.isel(time = index1)
met = met_ori.resample(time='10s').nearest()

_, index1 = np.unique(nav_ori['time'], return_index = True)
nav_ori = nav_ori.isel(time = index1)
nav = nav_ori.resample(time='10s').nearest()

#nav = nav_ori.resample(time='1s').nearest()
#met = met_ori

#%%
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

#_, index1 = np.unique(met['time'], return_index = True)
#reld_new1 = reld.sel(time=~reld.indexes['time'].duplicated(keep = 'first'))
#rels_new1 = rels.sel(time=~reld.indexes['time'].duplicated(keep='first'))
#rels_new = rels_new1.resample(time='1s').nearest()
#reld_new = reld_new1.resample(time='1s').nearest()


# Calculate winds based on method in the document denoted above
relsn = rels * np.cos(head + reld)
relse = rels * np.sin(head + reld)

sogn = sog * np.cos(cog)
soge = sog * np.sin(cog)

un = relsn - sogn
ue = relse - soge

dirt = np.mod(np.rad2deg(np.arctan2(ue, un)) + 360., 360)
ut = np.sqrt(un ** 2. + ue ** 2)

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
flag_cpc = exhaust[idx2] #'''This step is slow!!'''
#%%
index_cpc_mad = np.where(flag_cpc == 1)# pick up the contaminated time index
dirty2 = np.array(index_cpc_mad[0]) # name the index to be dirty(contaminated index)
index_cpc_clean_mad = np.where(flag_cpc == 0)
clean2 = np.array(index_cpc_clean_mad[0])# name the index to be clean(clean index)

#%%
cpc_con1 = np.ma.masked_where(qc_cpc != 0, cpc_con)
#win_cpc_df = pd.DataFrame({'wind_speed':ut, 'wind_direction':dirt, 'cpc_con':cpc_con1}, index = time_cpc)

win_cpc_df1 = pd.DataFrame({'wind_speed':ut[clean2], 'wind_direction':dirt[clean2], 'cpc_con':cpc_con1[clean2],'lat':nav['lat'][clean2]}, index = time_cpc[clean2])

#%% Plotting
from scipy.stats import gaussian_kde
x = win_cpc_df1['cpc_con']
y = win_cpc_df1['wind_speed']
fig, ax = plt.subplots()

#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#plt.scatter(win_cpc_df['cpc_con'][clean2],win_cpc_df['wind_speed'][clean2])
# a lot of overplotting
#ax.scatter(x, y, c=z, s=100, edgecolor='')
#ax.hexbin(x, y, gridsize=20, cmap=plt.cm.BuGn_r)
#im = ax.hexbin(y, x, gridsize=20, bins = 'log', 
#               cmap=plt.cm.BuGn)
im = ax.hexbin(y, x, gridsize=20, 
               cmap=plt.cm.BuGn)
#ax.set_xlim(0,2000)
# bins ='log' 
ax.set_ylabel('1/cc')
#axs[p].set_ylim((0,25))
ax.set_xlabel('m/s')
ax.set_title('CPC concentration & wind speed')
#ax.colorbar()

cb = fig.colorbar(im, ax=ax)
cb.set_label('counts')
#cb.set_label('log10(N)')
plt.show()

#%%
sns.set()
sns.jointplot(x = 'cpc_con' , y ='wind_speed', data = win_cpc_df1, color="purple")

# %%
# Jointplot - Scatterplot and Histogram
sns.set()
g = sns.jointplot(x = 'cpc_con' , y = 'wind_speed', data = win_cpc_df1, kind ="hex",xlim=[1,1800],ylim=[0.01,25], color="lightcoral")
#g = sns.jointplot(x = 'cpc_con' , y = 'wind_direction', data = win_cpc_df1, kind ="hex",xlim=[1,1800],color="lightcoral")
#data,xlim=[1,100],ylim=[0.01,100]
#ax.set_xscale('log')
#ax.set_yscale('log')
#g.ax_marg_x.set_xscale('log')
#g.ax_marg_y.set_yscale('log')
#g.ax_joint.set_xscale('log')
#g.ax_joint.set_yscale('log')
#%%
# Jointplot - Scatterplot and Histogram
sns.set()
sns.jointplot(x = 'cpc_con', y = 'wind_speed', data = win_cpc_df1,kind = "kde", color="purple") # contour plot