#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:19:01 2020
UHSAS-Wind
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
import seaborn as sns
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
datastream2 = 'maraosco'
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
#%% read in Jan 1st VAP
vap1 = netCDF4.Dataset('/Users/qingn/Downloads/Environmental VAP/MARCUS/V1.3/MARCUS_Environmental_Parameters_VAP_V1.3_Voyage3_20180116_20180304.cdf')


#%% Read in uhsas and wind
    
wind_3 = pd.read_csv('/Users/qingn/201803cpc_wind_lat_.csv',index_col = 0, parse_dates = True)
wind_2 = pd.read_csv('/Users/qingn/201802cpc_wind_lat_.csv',index_col = 0, parse_dates = True)
wind_1 = pd.read_csv('/Users/qingn/201801cpc_wind_lat_.csv',index_col = 0, parse_dates = True)
wind_12 = pd.read_csv('/Users/qingn/201712cpc_wind_lat_.csv',index_col = 0, parse_dates = True)
wind_11 = pd.read_csv('/Users/qingn/201711cpc_wind_lat_.csv',index_col = 0, parse_dates = True)

wind_total = pd.concat([wind_2,wind_3,wind_1,wind_11,wind_12])

del wind_2,wind_3,wind_1,wind_11,wind_12
# We try from Hobart to Macq first 0309-0325
#uhsas_ccn = pd.read_csv('/Users/qingn/ten_min_clean_cpc_aftms_accu.csv',index_col = 0, parse_dates = True)

list_date_four = ['2017-10-29','2017-12-03','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-25']

wind_total_10min = wind_total.resample('10min').mean()#
#%%
df_combine = pd.concat([ten_min,wind_total_10min], axis=1,join='inner')#
#%%
#hobart_casey= wind_total['2017-10-29':'2017-12-03']
#hobart_casey_10min = hobart_casey.resample('10min').mean()

#hobart_casey_aero = uhsas_ccn['2017-10-29':'2017-12-03']

#hobart_casey_combine = pd.concat([hobart_casey_aero, hobart_casey_10min], axis=1,join='inner')
hobart_casey_combine = df_combine['2017-10-29':'2017-12-03']
#
#hobart_davis= wind_total['2017-12-13':'2018-01-10']
#hobart_davis_10min = hobart_davis.resample('10min').mean()
#
#hobart_davis_aero = uhsas_ccn['2017-12-13':'2018-01-10']

#hobart_davis_combine = pd.concat([hobart_davis_aero, hobart_davis_10min], axis=1,join='inner')
hobart_davis_combine  = df_combine['2017-12-13':'2018-01-10']

#
#hobart_mawson= wind_total['2018-01-16':'2018-03-04']
#hobart_mawson_10min = hobart_davis.resample('10min').mean()
#
#hobart_mawson_aero = uhsas_ccn['2018-01-16':'2018-03-04']
#
#hobart_mawson_combine = pd.concat([hobart_davis_aero[:1829], hobart_davis_10min], axis=1)
hobart_mawson_combine = df_combine['2018-01-16':'2018-03-04']


#
#hobart_macq = wind_total['2018-03-09':'2018-03-25']
#hobart_macq_10min = hobart_macq.resample('10min').mean()
#
#hobart_macq_aero = uhsas_ccn['2018-03-09':'2018-03-25']
#
#hobar_macq_combine = pd.concat([hobart_macq_aero[:1829], hobart_macq_10min], axis=1)
hobart_macq_combine = df_combine['2018-03-09':'2018-03-25']


#%% Try Figure


#list_date_four = ['2017-10-29','2017-12-03','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-25']

clean_flag_davis = np.array([hobart_davis_combine['new_flag']])[0]
clean_flag_casey= np.array([hobart_casey_combine['new_flag']])[0]
clean_flag_mawson = np.array([hobart_mawson_combine['new_flag']])[0]
clean_flag_macq =np.array([hobart_macq_combine['new_flag']])[0]

voyages_combine = pd.concat([hobart_casey_combine[clean_flag_casey], hobart_davis_combine[clean_flag_davis],hobart_mawson_combine[clean_flag_mawson],hobart_macq_combine[clean_flag_macq]])
#%%

plt.scatter(hobart_davis_combine['accumulation'][clean_flag_davis],hobart_davis_combine['wind_speed'][clean_flag_davis])
plt.title('accumulation-wind_speed')
plt.xlabel('accumulation(#/cc)')
plt.ylabel('wind_speed(m/s)')
            
#plt.xlim([0,100])
#%%
sns.set(style="darkgrid")
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

#tips = sns.load_dataset("tips")
g = sns.jointplot("wind_speed","uhsas_large",  data=voyages_combine,
                  kind="reg", truncate=False, stat_func=r2,# xlim=(0, 100), ylim=(4, 20),
                  color="m", height=7)
g.set_axis_labels("wind speed", "uhsas_large(#/cc)", fontsize=26)
m = sns.jointplot("wind_speed","uhsas_middle",  data=voyages_combine,
                  kind="reg", truncate=False,stat_func=r2,# xlim=(0, 100), ylim=(4, 20),
                  color="r", height=7)
  #%%
m.set_axis_labels("wind speed", "uhsas_middle(#/cc)", fontsize=26)
n = sns.jointplot("wind_speed","uhsas_small",  data=voyages_combine,
                  kind="reg", truncate=False,ylim=(0, 300),# xlim=(0, 100), ,
                  color="b", height=7)
n.set_axis_labels("wind speed", "uhsas_small(#/cc)", fontsize=26)
      
o = sns.jointplot("wind_speed","uhsas",  data=voyages_combine,
                  kind="reg", truncate=False,ylim=(0, 500),# xlim=(0, 100), ,
                  color="b", height=7)
o.set_axis_labels("wind speed", "accumulation(#/cc)", fontsize=26)

           #%%
tips = sns.load_dataset("tips")
#%%
#import matplotlib.ticker as ticker
g = sns.jointplot(x=hobart_macq_combine[clean_flag_macq]["wind_speed"],
                  y=hobart_macq_combine[clean_flag_macq]["uhsas_small"], kind='kde',
                    ylim=[0,300])# ,xlim =[4,20]
sns.jointplot(x=hobart_macq_combine[clean_flag_macq]["wind_speed"],
                  y=hobart_macq_combine[clean_flag_macq]["uhsas_middle"], kind='kde',
                    )
sns.jointplot(x=hobart_macq_combine[clean_flag_macq]["wind_speed"],
                  y=hobart_macq_combine[clean_flag_macq]["uhsas_large"], kind='kde',
                    )
g.set_axis_labels("wind speed","accumulation_mode(#/cc)",  fontsize=26)
g.ax_marg_y.set_axis_off()
g.fig.suptitle('hobart_macq', y=1.08, fontsize=26)
#%%
g = sns.jointplot(x=hobart_davis_combine[clean_flag_davis]["accumulation"], y=hobart_davis_combine[clean_flag_davis]["wind_speed"], kind='kde',
                    xlim=[0,100],ylim =[4,20])
g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.ax_marg_y.set_axis_off()
g.fig.suptitle('hobart_davis', y=1.08, fontsize=26)
#%%
g = sns.jointplot(x=hobart_casey_combine[clean_flag_casey]["accumulation"], y=hobart_casey_combine[clean_flag_casey]["wind_speed"], kind='kde',
                    xlim=[0,150],ylim =[4,20])
g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.ax_marg_y.set_axis_off()
g.fig.suptitle('hobart_casey', y=1.08, fontsize=26)
#%%
g = sns.jointplot(x=hobart_mawson_combine[clean_flag_mawson]["accumulation"], y=hobart_mawson_combine[clean_flag_mawson]["wind_speed"], kind='kde',
                    xlim=[0,100],ylim =[4,20])
g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.fig.suptitle('hobart_mawson', y=1.08, fontsize=26)
#plt.title('hobart_mawson_combine',fontsize = '20')
#g.ax_joint.xaxis.set_major_formatter(ticker.MultipleLocator(45))

#plt.tick_params(axis="both", labelsize=28)
#lm = lm.annotate(stats.pearsonr, fontsize=28)
#ax= lm.axes
#ax.set_xlim([0,150])
#%%

g = sns.jointplot(x=hobart_mawson_combine[clean_flag_mawson]["accumulation"], y=hobart_mawson_combine[clean_flag_mawson]["wind_direction"], kind='kde')

#g = sns.jointplot(x=hobart_macq_combine[clean_flag_macq]["accumulation"], y=hobart_macq_combine[clean_flag_macq]["wind_speed"], kind='kde')
#                    ,xlim=[0,100],ylim =[4,20])
#g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.set_axis_labels("accumulation_mode(#/cc)", "wind direction", fontsize=26)
g.ax_marg_y.set_axis_off()
g.fig.suptitle('hobart_macq', y=1.08, fontsize=26)

#%%
#g = sns.jointplot(x=hobart_davis_combine[clean_flag_davis]["accumulation"], y=hobart_davis_combine[clean_flag_davis]["wind_speed"], kind='kde')
#                    ,xlim=[0,100],ylim =[4,20])
g = sns.jointplot(x=hobart_mawson_combine[clean_flag_mawson]["accumulation"], y=hobart_mawson_combine[clean_flag_mawson]["wind_direction"], kind='kde')
#g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.set_axis_labels("accumulation_mode(#/cc)", "wind direction", fontsize=26)
g.fig.suptitle('hobart_davis', y=1.08, fontsize=26)

#%%
#g = sns.jointplot(x=hobart_casey_combine[clean_flag_casey]["accumulation"], y=hobart_casey_combine[clean_flag_casey]["wind_speed"], kind='kde')
#                  ,xlim=[0,100],ylim =[4,20])
g = sns.jointplot(x=hobart_mawson_combine[clean_flag_mawson]["accumulation"], y=hobart_mawson_combine[clean_flag_mawson]["wind_direction"], kind='kde')

#g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.set_axis_labels("accumulation_mode(#/cc)", "wind direction", fontsize=26)
g.fig.suptitle('hobart_casey', y=1.08, fontsize=26)

#%%
#g = sns.jointplot(x=hobart_mawson_combine[clean_flag_mawson]["accumulation"], y=hobart_mawson_combine[clean_flag_mawson]["wind_speed"], kind='kde')
g = sns.jointplot(x=hobart_mawson_combine[clean_flag_mawson]["uhsas"], y=hobart_mawson_combine[clean_flag_mawson]["wind_direction"], kind='kde')
#                    ,xlim=[0,100],ylim =[4,20])
#g.set_axis_labels("accumulation_mode(#/cc)", "wind speed", fontsize=26)
g.set_axis_labels("accumulation_mode(#/cc)", "wind direction", fontsize=26)

g.fig.suptitle('hobart_mawson', y=1.08, fontsize=26)
g.fig.tight_layout()
ax = plt.gca()
#ax.text(0.0, 0.1, "NullFormatter()", fontsize=16, transform=ax.transAxes)
#ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.tick_params(which='major', width=1.00, length=3,labelsize = 10)