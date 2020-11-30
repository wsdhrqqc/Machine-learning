#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 07:18:28 2020

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
#%% read file
df = pd.read_csv('/Users/qingn/ten_min_clean_cpc_aftms_accu.csv', parse_dates=True)
uhsas_99 = pd.read_csv('uhsas_99_con.csv',parse_dates = True)
met = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201*.nc','10min')
uhsas = arm_read_netcdf_for_time_resolution('/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201*.nc','10s')
wind = pd.read_csv('/Users/qingn/four_voyage_env_ccn.csv')
wind_1_cpc = pd.read_csv('/Users/qingn/201711cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_2_cpc = pd.read_csv('/Users/qingn/201712cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_3_cpc = pd.read_csv('/Users/qingn/201801cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_4_cpc = pd.read_csv('/Users/qingn/201802cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_5_cpc = pd.read_csv('/Users/qingn/201803cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)

wind_full = pd.concat([wind_1_cpc,wind_2_cpc,wind_3_cpc,wind_4_cpc,wind_5_cpc])
wind_full_10min = wind_full.resample('10T').mean()

groups = wind_full.groupby(Grouper(freq = 'D'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups],axis=1)
slf_time = pd.date_range(start ='2017-10-29',end = '2018-03-24', freq='D')# All together 147 days
#months.columns  = slf_time
error = months.std()
err = np.array(error)
#%%
list_date_four = ['2017-10-02','2017-12-03','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-24']
psd = uhsas['size_distribution']
#rain = met['rain_intensity']
lower_limit = uhsas['lower_size_limit'][0]
psd_new = psd[:-17]
rain_mount = met['rain_intensity'][24:21097]

psd_time = pd.to_datetime(psd.time.values)[:-17]
#%% UHSAS


#%%

xx  = rain_mount>0.01
xx = xx.astype(int)
df_new = df[24:]
df_new['rain'] = rain_mount
df_new['rain_flag'] = xx
df_new['over_ocean'] = 0
#%%
df_new.set_index(['Unnamed: 0'], inplace = True)
df_new.index = pd.to_datetime(df_new.index)

df_new['over_ocean'].loc['2017-10-29':'2017-12-03'] = 1
df_new['over_ocean'].loc['2017-12-13':'2018-01-10'] = 1
df_new['over_ocean'].loc['2018-01-16':'2018-03-04'] = 1
df_new['over_ocean'].loc['2018-03-09':'2018-03-24'] = 1
df_new['location_flag'] = df_new['over_ocean']==1
df_new['location_flag'] = df_new['location_flag'].astype(int)
#%%
df_new['sum_flag'] = df_new['new_flag'].astype(int)
df_new['sum_flag'][np.where(df_new['rain_flag'] == 1)[0]] = 2

aero_wind = pd.merge(df_new, wind_full_10min,left_index = True, right_index = True, how = 'outer')

ss0_1 = np.where(df_new['SS']==0.1)[0]
ss0_5 = np.where(df_new['SS']==0.5)[0]
ss0_2 = np.where(df_new['SS']==0.2)[0]
ss0_8 = np.where(df_new['SS']==0.8)[0]
ss0_10 = np.where(df_new['SS']==1)[0]
# sum =1 is clean
#%% Coarse aero data analysis
coarse_array = uhsas['concentration'].values
coarse = [np.nansum(coarse_array[i,75:]) for i in range(1265400)]
df_uhsass = pd.DataFrame(coarse)
df_uhsass['time'] = uhsas['concentration'].time
df_uhsass = df_uhsass.set_index('time')
df_uhsass = df_uhsass.rename({0: 'coarse'}, axis='columns')
df_10min_uhsas = df_uhsass.groupby(Grouper(freq='10T')).mean()
df_coarse_wind = pd.concat([df_wind_aero,df_10min_uhsas],axis =1, sort = False)
#%% Combine WIBS4
#%
wibs = pd.read_csv('/Users/qingn/Desktop/NQ/WIBS_MARCUS_NC_3std_noport.csv', parse_dates=True)
exhaust_wibs = pd.read_csv('/Users/qingn/Desktop/NQ/WIBS_MARCUS_NC_3std_noport_exhausfilter.csv', parse_dates=True)
ex_time1=pd.to_datetime(exhaust_wibs['Date_Time'], format = '%d.%m.%Y %H:%M:%S')
#date_time1= pd.to_datetime(wibs['Date_Time'], format = '%d.%m.%Y %H:%M:%S')

df_wibs = exhaust_wibs[['FBAP','All','NonF']]
df_wibs['time'] = ex_time1
df_wibs = df_wibs.set_index('time')
df_wibs = df_wibs.loc[~df_wibs.index.duplicated(keep='first')]
df_wibs['time'] = df_wibs.index
df_try_10min = pd.merge_asof(df_coarse_wind, df_wibs,on = 'time',tolerance = pd.Timedelta('10min'),direction='nearest')#,allow_exact_matches = False
df_try_10min = df_try_10min.set_index('time')


df_ori_wibs = wibs[['FBAP','All','NonF']]
df_ori_wibs['time'] = ex_time1
df_ori_wibs = df_ori_wibs.set_index('time')

#df_ori_wibs = df_ori_wibs.loc[~df_ori_wibs.index.duplicated(keep='first')]
#df_ori_wibs['time'] = df_ori_wibs.index
df_ori_try_10min = pd.merge_asof(df_ori_wibs,df_10min['new_flag'][ss0_2],on = 'time',tolerance = pd.Timedelta('9min'),direction='backward')#,allow_exact_matches = False
df_ori_try_10min_ = pd.merge_asof(df_coarse_wind,df_ori_try_10min,on ='time',tolerance = pd.Timedelta('9min'),direction='backward')
df_ori_try_10min.set_index('time')
#df_ori_try_10min = df_ori_try_10min.set_index('time')
#df_wibs = pd.DataFrame({'time':ex_time1,'All':exhaust_wibs['All'],'FBAP':exhaust_wibs['FBAP'],'NonF':exhaust_wibs['NonF']})
#%%
fig, ax = plt.subplots(nrows=2, ncols=1, sharey=False, figsize=(12, 8))    
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
#ax = ax0.twinx()

ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_2, np.where(df_new['sum_flag']==1)[0])][list_date_four[0]:list_date_four[1]],'r.',label = '0.2%')
ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_5, np.where(df_new['sum_flag']==1)[0])][list_date_four[0]:list_date_four[1]],'b.',label = '0.5%')
ax[0].legend()
ax[0].set_ylabel('N_CCN(#/cc)')
pcm = ax[0].pcolorfast(ax[0].get_xlim(), ax[0].get_ylim(),
              df_new[list_date_four[0]:list_date_four[1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs = ax[1].twinx()
ax[1].plot(df_wibs['All'][list_date_four[0]:list_date_four[1]],'b^',label='All')
axs.plot(df_wibs['FBAP'][list_date_four[0]:list_date_four[1]],'g*',label='FBAP')

pcm = ax[1].pcolorfast(ax[1].get_xlim(), ax[1].get_ylim(),
              df_new[list_date_four[0]:list_date_four[1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
ax[1].legend()

axs.legend()
axs.set_ylim([0,40])
ax[1].set_ylabel('All(#/L)')
ax[1].set_ylim([0,1000])
axs.set_ylabel('FBAP(#/L)')
#cbar.set_label('(1/cc)', rotation=90)
fig.autofmt_xdate()
plt.title('V1')


#%%
fig, ax = plt.subplots(nrows=2, ncols=1, sharey=False, figsize=(12, 8))    
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
#ax = ax0.twinx()

ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_2, np.where(df_new['sum_flag']==1)[0])][list_date_four[2]:list_date_four[3]],'r.',label = '0.2%')
ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_5, np.where(df_new['sum_flag']==1)[0])][list_date_four[2]:list_date_four[3]],'b.',label = '0.5%')
ax[0].legend()
ax[0].set_ylabel('N_CCN(#/cc)')
pcm = ax[0].pcolorfast(ax[0].get_xlim(), ax[0].get_ylim(),
              df_new[list_date_four[2]:list_date_four[3]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs = ax[1].twinx()
ax[1].plot(df_wibs['All'][list_date_four[2]:list_date_four[3]],'b^',label='All')
axs.plot(df_wibs['FBAP'][list_date_four[2]:list_date_four[3]],'g*',label='FBAP')

pcm = ax[1].pcolorfast(ax[1].get_xlim(), ax[1].get_ylim(),
              df_new[list_date_four[2]:list_date_four[3]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
ax[1].legend()
axs.legend()

axs.set_ylim([0,40])
ax[1].set_ylabel('All(#/L)')
axs.set_ylabel('FBAP(#/L)')
#fig.colorbar(pcm, ax=axs, orientation='horizontal', fraction=.1)
#cbar=fig.colorbar(pcm, ax=axs)
#cbar.set_label('(1/cc)', rotation=90)
fig.autofmt_xdate()
fig.tight_layout()
plt.title('V2')
#%%


fig, ax = plt.subplots(nrows=2, ncols=1, sharey=False, figsize=(12, 8))    
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
#ax = ax0.twinx()

ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_2, np.where(df_new['sum_flag']==1)[0])][list_date_four[4]:list_date_four[5]],'r.',label = '0.2%')
ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_5, np.where(df_new['sum_flag']==1)[0])][list_date_four[4]:list_date_four[5]],'b.',label = '0.5%')
ax[0].legend()
ax[0].set_ylabel('N_CCN(#/cc)')
pcm = ax[0].pcolorfast(ax[0].get_xlim(), ax[0].get_ylim(),
              df_new[list_date_four[4]:list_date_four[5]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs = ax[1].twinx()
ax[1].plot(df_wibs['All'][list_date_four[4]:list_date_four[5]],'b^',label='All')
axs.plot(df_wibs['FBAP'][list_date_four[4]:list_date_four[5]],'g*',label='FBAP')

pcm = ax[1].pcolorfast(ax[1].get_xlim(), ax[1].get_ylim(),
              df_new[list_date_four[4]:list_date_four[5]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
ax[1].legend()
axs.legend()
ax[1].set_ylabel('All(#/L)')
axs.set_ylabel('FBAP(#/L)')
fig.autofmt_xdate()
plt.title('V3')


#%%

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 8))    
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
#ax = ax0.twinx()

ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_2, np.where(df_new['sum_flag']==1)[0])][list_date_four[6]:list_date_four[7]],'r.',label = '0.2%')
ax[0].plot(df_new['N_CCN'][np.intersect1d(ss0_5, np.where(df_new['sum_flag']==1)[0])][list_date_four[6]:list_date_four[7]],'b.',label = '0.5%')
ax[0].legend()
ax[0].set_ylabel('N_CCN(#/cc)')
pcm = ax[0].pcolorfast(ax[0].get_xlim(), ax[0].get_ylim(),
              df_new[list_date_four[6]:list_date_four[7]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs = ax[1].twinx()
ax[1].plot(df_wibs['All'][list_date_four[6]:list_date_four[7]],'b*',label='All')
axs.plot(df_wibs['FBAP'][list_date_four[6]:list_date_four[7]],'g^',label='FBAP')

pcm = ax[1].pcolorfast(ax[1].get_xlim(), ax[1].get_ylim(),
              df_new[list_date_four[6]:list_date_four[7]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
ax[1].legend()
axs.legend()
axs.set_ylim([0,100])
ax[1].set_ylabel('All(#/L)')
axs.set_ylabel('FBAP(#/L)')
fig.autofmt_xdate()
plt.title('V4')
#%%
fig = plt.figure()
plt.plot(df_wibs['FBAP'],'g*',label = 'FBAP')
plt.plot(df_wibs['All'],'b^',label = 'All')
plt.legend()
fig.autofmt_xdate()
plt.title('WIBS4')
plt.ylabel('(#/L)')

#%%

fig, axs = plt.subplots(2, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*2],sharex=False)
pcm = axs[0].pcolorfast(axs[0].get_xlim(), (0,2500),
              df_new[list_date_four[0]:list_date_four[1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[0].plot(df_new['N_CCN'][ss0_5][list_date_four[0]:list_date_four[1]],'r.',label = '0.5%')
axs[0].plot(df_new['N_CCN'][ss0_2][list_date_four[0]:list_date_four[1]],'g-',label = '0.2%')
#axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[0]:list_date_four[1]],'k.',label = '0.8%')
axs[0].set_ylim((0,500))
axs[0].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[0])

axs[0].legend(framealpha=0.5,markerscale = 2.)
axs[0].set_title('Hobart-Casey')
#plt.setp(axs[0].get_xticklabels(), rotation=30, horizontalalignment='right')
#fig.autofmt_xdate()
#plt.show()
#%%
fig = plt.figure(figsize = [FIGWIDTH,FIGHEIGHT])

ax = plt.gca()

pcm = ax.pcolormesh(psd[4000:6000].time,lower_limit,psd[4000:6000].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
#pcm = ax.pcolormesh(dn_dlogD['time'],lower_bd/1000,dn_dlogD.T,norm=matplotlib.colors.LogNorm(),
#                   cmap='jet')
#plasma,magma [4320:10800]
cbar= fig.colorbar(pcm, ax=ax)
cbar.set_label('(1/cc)', rotation=90)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.xaxis_date()
plt.ylabel('Dp(um)')
fig.autofmt_xdate()
plt.title('UHSAS clean')
fig.tight_layout() 
#%%
fig = plt.figure(figsize = [FIGWIDTH,FIGHEIGHT])


plt.plot(df_new[list_date_four[0]:list_date_four[1]].rain,'r.')# ax4.get_ylim()
zeros = df_new[df_new['rain_flag'] ==0]
#pcm = ax4.pcolorfast(df_new[list_date_four[0]:list_date_four[1]]['rain_flag'].values[np.newaxis],vmin = min, vmax = max,
#                            cmap='RdYlGn', alpha=0.3)
ax = df_new[list_date_four[0]:list_date_four[1]].rain_flag.plot()
for x in zeros:
    ax.axvline(df_new.index[x],color = 'y',linewidth=5,alpha = 0.03)

#pcm = ax4.pcolorfast(df_new[list_date_four[0]:list_date_four[1]].index,df_new[list_date_four[0]:list_date_four[1]].rain,df_new[list_date_four[0]:list_date_four[1]]['rain_flag'].values[np.newaxis],
#                            cmap='RdYlGn', alpha=0.3)
cbar= fig.colorbar(pcm, ax=ax)
#%%
fig = plt.figure(figsize = [FIGWIDTH,FIGHEIGHT])
plt.plot(met.time,met['rain_intensity'],'r.')
plt.yscale('log')
#%%

#list_date_four[0]:list_date_four[1]
fig, axs = plt.subplots(3, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*3],sharex=True)
#plt.subplot(211)
pcm = axs[0].pcolormesh(psd.loc[list_date_four[0]:list_date_four[1]].time,lower_limit,psd.loc[list_date_four[0]:list_date_four[1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
#pcm = ax.pcolormesh(dn_dlogD['time'],lower_bd/1000,dn_dlogD.T,norm=matplotlib.colors.LogNorm(),
#                   cmap='jet')
#plt.ylabel('nm')
#plasma,magma [4320:10800]
axs[0].set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
axs[0].set_yscale('log')
axs[0].plot(df_new['N_CCN'][ss0_5][list_date_four[0]:list_date_four[1]],'r.',label = '0.5%')
#yy = axs[0].get_ylim()[1]
axs[0].pcolorfast(axs[0].get_xlim(), (10,2000),
              df_new[list_date_four[0]:list_date_four[1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)

#fig.colorbar(pcm, ax=axs[0], orientation='horizontal')#fig.colorbar(pcm, ax=axs[0])
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)

#axs.xaxis_date()
#ax[0].set_ylabel('Dp(um)')
#axs[1] = df_new[list_date_four[0]:list_date_four[1]].rain_flag.plot()
#plt.title('UHSAS clean')

#plt.plot(t, s)
#plt.subplot(212)

#axs[1].plot(df_new[list_date_four[0]:list_date_four[1]].index,df_new[list_date_four[0]:list_date_four[1]].rain,'r.')

pcm = axs[2].pcolorfast(axs[2].get_xlim(), (0,100),
              df_new[list_date_four[0]:list_date_four[1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[2].plot(df_new[list_date_four[0]:list_date_four[1]].rain,'b-',label = 'rain intensity')
axs[2].set_yscale('log')
axs[2].set_ylabel('rain_intensity(mm/h)',color = 'blue')
axs[2].tick_params(axis='y', labelcolor='blue')
axs4 = axs[2].twinx()
color = 'tab:red'
axs4.set_ylabel('wind_Speed(m/s)', color=color)  # we already handled the x-label with ax1
#ax2.plot(t, data2, color=color)
axs4.tick_params(axis='y', labelcolor=color)
axs4.plot(aero_wind[list_date_four[0]:list_date_four[1]].wind_speed,'r-',label = 'wind_speed')

axs4.errorbar(slf_time[4:36],months.mean()[4:36],err[4:36],fmt ='o',color = 'black',ecolor = 'gray',elinewidth = 3,capsize = 0)
#axs[1].legend()
#axs4.legend()
#months[:30].boxplot(showfliers =False)
#plt.show()
#cbar.set_label('(1/cc)', rotation=9

#divider = make_axes_locatable(ax)
#cax = divider.append_axes('top', size='5%', pad=0.05)

#fig.colorbar(pcm, ax = axs[1] , orientation='horizontal')
#x = range(len(df_new[list_date_four[0]:list_date_four[1]].rain))
#y = range(20)
#xc,yy = np.meshgrid(x,y)
#im = axs[1].pcolorfast(xc, yy,
#                        df_new[list_date_four[0]:list_date_four[1]]['rain_flag'].values[np.newaxis],
#                            cmap='RdYlGn', alpha=0.3)#
#im = axs[1].pcolorfast(axs[1].get_xlim(), axs[1].get_ylim(),
#                        df_new[list_date_four[0]:list_date_four[1]]['rain_flag'].values[np.newaxis],
#                            cmap='RdYlGn', alpha=0.3)#

#fig.colorbar(im, pad = 0.1)#, orientation='horizontal')


#fig.tight_layout() 
pcm = axs[1].pcolorfast(axs[1].get_xlim(), (0,2500),
              df_new[list_date_four[0]:list_date_four[1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[1].plot(df_new['N_CCN'][ss0_5][list_date_four[0]:list_date_four[1]],'r.',label = '0.5%')
axs[1].plot(df_new['N_CCN'][ss0_2][list_date_four[0]:list_date_four[1]],'g-',label = '0.2%')
axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[0]:list_date_four[1]],'k.',label = '0.8%')
axs[1].set_ylim((0,500))
axs[1].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[1])

axs[1].legend(framealpha=0.5,markerscale = 2.)
axs[0].set_title('Hobart-Casey')
plt.setp(axs[2].get_xticklabels(), rotation=30, horizontalalignment='right')
fig.autofmt_xdate()
#plt.close()
#fig.tight_layout
#%%
i = 0
fig, axs = plt.subplots(3, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*3],sharex=True)

pcm = axs[0].pcolormesh(psd.loc[list_date_four[i]:list_date_four[i+1]].time,lower_limit,psd.loc[list_date_four[i]:list_date_four[i+1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')

axs[0].set_title('Hobart-Macquarie')
axs[0].set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
axs[0].set_yscale('log')

axs[0].pcolorfast(axs[0].get_xlim(), (40,1500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)

pcm = axs[1].pcolorfast(axs[0].get_xlim(), (0,2500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[1].plot(df_new['N_CCN'][ss0_5][list_date_four[i]:list_date_four[i+1]],'r-',label = '0.5%')
axs[1].plot(df_new['N_CCN'][ss0_2][list_date_four[i]:list_date_four[i+1]],'g.',label = '0.2%')
axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[i]:list_date_four[i+1]],'k.',label = '0.8%')
axs[1].set_ylim((0,850))
axs[1].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[1])
axs[1].legend(framealpha=0.5,markerscale = 2.)


pcm = axs[2].pcolorfast(axs[0].get_xlim(), (0,100),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[2].plot(df_new[list_date_four[i]:list_date_four[i+1]].rain,'b.',label = 'rain intensity')
axs[2].set_yscale('log')
axs[2].set_ylabel('rain_intensity(mm/h)',color = 'blue')
axs[2].tick_params(axis='y', labelcolor='blue')
axs4 = axs[2].twinx()
color = 'tab:red'
axs4.set_ylabel('wind_Speed(m/s)', color=color)  # we already handled the x-label with ax1
#ax2.plot(t, data2, color=color)
axs4.tick_params(axis='y', labelcolor=color)
axs4.plot(aero_wind[list_date_four[i]:list_date_four[i+1]].wind_speed,'r-',label = 'wind_speed')

axs4.errorbar(slf_time[4:36],months.mean()[4:36],err[4:36],fmt ='o',color = 'black',ecolor = 'gray',elinewidth = 3,capsize = 0)

plt.setp(axs[2].get_xticklabels(), rotation=30, horizontalalignment='right')
fig.autofmt_xdate()
#%% change a little bit
i = 0
fig, axs = plt.subplots(3, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*3],sharex=True)

pcm = axs[0].pcolormesh(psd.loc[list_date_four[i]:list_date_four[i+1]].time,lower_limit,psd.loc[list_date_four[i]:list_date_four[i+1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')

axs[0].set_title('Hobart-Macquarie')
axs[0].set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
axs[0].set_yscale('log')
axs[0].plot(df_new[list_date_four[i]:list_date_four[i+1]].rain*5,'r-',label = 'rain intensity')
pcm1 = axs[0].pcolorfast(axs[0].get_xlim(), (1,2500),#(40,1500)
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)

pcm = axs[1].pcolorfast(axs[0].get_xlim(), (0,2500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[1].plot(df_new['N_CCN'][ss0_5][list_date_four[i]:list_date_four[i+1]],'r-',label = '0.5%')
axs[1].plot(df_new['N_CCN'][ss0_2][list_date_four[i]:list_date_four[i+1]],'g.',label = '0.2%')
axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[i]:list_date_four[i+1]],'k.',label = '0.8%')
axs[1].set_ylim((0,850))
axs[1].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[1])
axs[1].legend(framealpha=0.5,markerscale = 2.)


pcm = axs[2].pcolorfast(axs[0].get_xlim(), (0,100),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[2].plot(df_new[list_date_four[i]:list_date_four[i+1]].rain,'b.',label = 'rain intensity')
axs[2].set_yscale('log')
axs[2].set_ylabel('rain_intensity(mm/h)',color = 'blue')
axs[2].tick_params(axis='y', labelcolor='blue')
axs4 = axs[2].twinx()
color = 'tab:red'
axs4.set_ylabel('wind_Speed(m/s)', color=color)  # we already handled the x-label with ax1
#ax2.plot(t, data2, color=color)
axs4.tick_params(axis='y', labelcolor=color)
axs4.plot(aero_wind[list_date_four[i]:list_date_four[i+1]].wind_speed,'r-',label = 'wind_speed')

axs4.errorbar(slf_time[4:36],months.mean()[4:36],err[4:36],fmt ='o',color = 'black',ecolor = 'gray',elinewidth = 3,capsize = 0)

plt.setp(axs[2].get_xticklabels(), rotation=30, horizontalalignment='right')
fig.autofmt_xdate()
#%% Just see how the precipitation change size distribution
plt.figure(figsize  = (FIGWIDTH,FIGHEIGHT))
# plt.plot()
ax1 = plt.gca()
pcm = ax1.pcolormesh(psd.loc[list_date_four[i]:list_date_four[i+1]].time,lower_limit,psd.loc[list_date_four[i]:list_date_four[i+1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')

ax1.set_title('Hobart-Macquarie')
ax1.set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
ax1.set_yscale('log')
ax1.plot(df_new[list_date_four[i]:list_date_four[i+1]].rain*5,'r-',label = 'rain intensity')
pcm1 = ax1.pcolorfast(ax1.get_xlim(), (1,2500),#(40,1500)
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)



#%%
i = 2
fig, axs = plt.subplots(3, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*3],sharex=True)
pcm = axs[0].pcolormesh(psd.loc[list_date_four[i]:list_date_four[i+1]].time,lower_limit,psd.loc[list_date_four[i]:list_date_four[i+1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
axs[0].set_title('Hobart-Davis')
axs[0].set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
axs[0].set_yscale('log')
axs[0].pcolorfast(axs[0].get_xlim(), (40,1500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)

pcm = axs[1].pcolorfast(axs[1].get_xlim(), (0,2500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[1].plot(df_new['N_CCN'][ss0_5][list_date_four[i]:list_date_four[i+1]],'r-',label = '0.5%')
axs[1].plot(df_new['N_CCN'][ss0_2][list_date_four[i]:list_date_four[i+1]],'g.',label = '0.2%')
axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[i]:list_date_four[i+1]],'k.',label = '0.8%')
axs[1].set_ylim((0,500))
axs[1].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[1])
axs[1].legend(framealpha=0.5,markerscale = 2.)

pcm = axs[2].pcolorfast(axs[1].get_xlim(), (0,100),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[2].plot(df_new[list_date_four[i]:list_date_four[i+1]].rain,'b-',label = 'rain intensity')
axs[2].set_yscale('log')
axs[2].set_ylabel('rain_intensity(mm/h)',color = 'blue')
axs[2].tick_params(axis='y', labelcolor='blue')
axs4 = axs[2].twinx()
color = 'tab:red'
axs4.set_ylabel('wind_Speed(m/s)', color=color)  # we already handled the x-label with ax1
#ax2.plot(t, data2, color=color)
axs4.tick_params(axis='y', labelcolor=color)
axs4.plot(aero_wind[list_date_four[i]:list_date_four[i+1]].wind_speed,'r-',label = 'wind_speed')

axs4.errorbar(slf_time[45:74],months.mean()[45:74],err[45:74],fmt ='o',color = 'black',ecolor = 'gray',elinewidth = 3,capsize = 0)

plt.setp(axs[2].get_xticklabels(), rotation=30, horizontalalignment='right')
fig.autofmt_xdate()
#%%
i = 4
fig, axs = plt.subplots(3, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*3],sharex=True)
pcm = axs[0].pcolormesh(psd.loc[list_date_four[i]:list_date_four[i+1]].time,lower_limit,psd.loc[list_date_four[i]:list_date_four[i+1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
axs[0].set_title('Hobart-Mawson')
axs[0].set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
axs[0].set_yscale('log')
axs[0].pcolorfast(axs[0].get_xlim(), (40,1500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)

pcm = axs[1].pcolorfast(axs[1].get_xlim(), (0,2500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[1].plot(df_new['N_CCN'][ss0_5][list_date_four[i]:list_date_four[i+1]],'r-',label = '0.5%')
axs[1].plot(df_new['N_CCN'][ss0_2][list_date_four[i]:list_date_four[i+1]],'g.',label = '0.2%')
axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[i]:list_date_four[i+1]],'k.',label = '0.8%')
axs[1].set_ylim((0,500))
axs[1].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[1])
axs[1].legend(framealpha=0.5,markerscale = 2.)

pcm = axs[2].pcolorfast(axs[1].get_xlim(), (0,100),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[2].plot(df_new[list_date_four[i]:list_date_four[i+1]].rain,'b-',label = 'rain intensity')
axs[2].set_yscale('log')
axs[2].set_ylabel('rain_intensity(mm/h)',color = 'blue')
axs[2].tick_params(axis='y', labelcolor='blue')
axs4 = axs[2].twinx()
color = 'tab:red'
axs4.set_ylabel('wind_Speed(m/s)', color=color)  # we already handled the x-label with ax1
#ax2.plot(t, data2, color=color)
axs4.tick_params(axis='y', labelcolor=color)
axs4.plot(aero_wind[list_date_four[i]:list_date_four[i+1]].wind_speed,'r-',label = 'wind_speed')

axs4.errorbar(slf_time[79:127],months.mean()[79:127],err[79:127],fmt ='o',color = 'black',ecolor = 'gray',elinewidth = 3,capsize = 0)

plt.setp(axs[2].get_xticklabels(), rotation=30, horizontalalignment='right')
fig.autofmt_xdate()
#%%
i = 6
fig, axs = plt.subplots(3, constrained_layout=True,figsize = [FIGWIDTH,FIGHEIGHT*3],sharex=True)
pcm = axs[0].pcolormesh(psd.loc[list_date_four[i]:list_date_four[i+1]].time,lower_limit,psd.loc[list_date_four[i]:list_date_four[i+1]].T,norm=matplotlib.colors.LogNorm(),
                   cmap='jet')
axs[0].set_title('Hobart-Macquarie')
axs[0].set_ylabel('size_distribution')
#ax[0].colorbar(pcm)
axs[0].set_yscale('log')
axs[0].pcolorfast(axs[0].get_xlim(), (40,1500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
cbar=fig.colorbar(pcm, ax=axs[0])
cbar.set_label('(1/cc)', rotation=90)

pcm = axs[1].pcolorfast(axs[1].get_xlim(), (0,2500),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[1].plot(df_new['N_CCN'][ss0_5][list_date_four[i]:list_date_four[i+1]],'r-',label = '0.5%')
axs[1].plot(df_new['N_CCN'][ss0_2][list_date_four[i]:list_date_four[i+1]],'g.',label = '0.2%')
axs[1].plot(df_new['N_CCN'][ss0_8][list_date_four[i]:list_date_four[i+1]],'k.',label = '0.8%')
axs[1].set_ylim((0,850))
axs[1].set_ylabel('NCCN(1/cc)')
cbar=fig.colorbar(pcm, ax=axs[1])
axs[1].legend(framealpha=0.5,markerscale = 2.)

pcm = axs[2].pcolorfast(axs[1].get_xlim(), (0,100),
              df_new[list_date_four[i]:list_date_four[i+1]]['sum_flag'].values[np.newaxis],
              cmap='RdYlGn', alpha=0.3)
axs[2].plot(df_new[list_date_four[i]:list_date_four[i+1]].rain,'b.',label = 'rain intensity')
axs[2].set_yscale('log')
axs[2].set_ylabel('rain_intensity(mm/h)',color = 'blue')
axs[2].tick_params(axis='y', labelcolor='blue')
axs4 = axs[2].twinx()
color = 'tab:red'
axs4.set_ylabel('wind_Speed(m/s)', color=color)  # we already handled the x-label with ax1
#ax2.plot(t, data2, color=color)
axs4.tick_params(axis='y', labelcolor=color)
axs4.plot(aero_wind[list_date_four[i]:list_date_four[i+1]].wind_speed,'r-',label = 'wind_speed')

axs4.errorbar(slf_time[131:],months.mean()[131:],err[131:],fmt ='o',color = 'black',ecolor = 'gray',elinewidth = 3,capsize = 0)

plt.setp(axs[2].get_xticklabels(), rotation=30, horizontalalignment='right')
fig.autofmt_xdate()
#%% Monthly averale spectra CCN 

ccn_series = [df_new.iloc[np.intersect1d(small_scaler,ss0_1)].groupby(Grouper(freq='M'))['N_CCN'],
                          df_new.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['N_CCN'],
                          df_new.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['N_CCN'],
                          df_new.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['N_CCN'],
                          df_new.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['N_CCN']]
#
#plt.errorbar(ccn_series[0].mean().index,ccn_series[0].mean(),yerr = ccn_series[0].std(),fmt = '-o',uplims = True, lolims = True,label='0.1%')
plt.errorbar(ccn_series[1].mean().index,ccn_series[1].mean(),yerr = ccn_series[1].std(),fmt = ':o',uplims = True, lolims = True,label='0.2%')
plt.errorbar(ccn_series[2].mean().index,ccn_series[2].mean(),yerr = ccn_series[2].std(),label='0.5%')
plt.errorbar(ccn_series[3].mean().index,ccn_series[3].mean(),yerr = ccn_series[3].std(),uplims = True, lolims = True,label='0.8%')
plt.errorbar(ccn_series[4].mean().index,ccn_series[4].mean(),yerr = ccn_series[4].std(),marker = '*',label='1.0%',uplims = True, lolims = True)
plt.legend()
plt.ylabel('NCCN(#/cc)')
plt.title('monthly N_CCN')
#%%
clean_small = np.intersect1d(small_scaler,np.where(df_wind_aero['new_flag']==True))

clean_coarse_series = [df_coarse_wind.iloc[clean_small].groupby(Grouper(freq = '10D'))['coarse']]
#%%
cpc_series = [df_wind_aero.iloc[np.where(df_wind_aero['new_flag']==True)[0]].groupby(Grouper(freq='10D'))['cpc_con']]
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['cpc_con']]
#
plt.errorbar(cpc_series[0].mean().index,cpc_series[0].mean(),yerr = cpc_series[0].std(),fmt = '-o',uplims = True, lolims = True,label='0.1%')
#plt.errorbar(cpc_series[0].mean().index,cpc_series[0].mean(),yerr = cpc_series[0].std(),fmt = ':o',uplims = True, lolims = True,label='0.2%')
#plt.errorbar(cpc_series[2].mean().index,ccn_series[2].mean(),yerr = ccn_series[2].std(),label='0.5%')
#plt.errorbar(cpc_series[3].mean().index,ccn_series[3].mean(),yerr = ccn_series[3].std(),uplims = True, lolims = True,label='0.8%')
#plt.errorbar(cpc_series[4].mean().index,ccn_series[4].mean(),yerr = ccn_series[4].std(),marker = '*',label='1.0%',uplims = True, lolims = True)
#plt.legend()
#plt.ylabel('CN(#/cc)')
plt.title('10D_avrg cpc_con')
#plt.tight_layout()
#fig.autofmt_xdate()
#%%coarse
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
coarse_series = [df_coarse_wind.iloc[small_scaler].groupby(Grouper(freq='10D'))['coarse']]
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['cpc_con']]
#
#plt.errorbar(ccn_series[0].mean().index,ccn_series[0].mean(),yerr = ccn_series[0].std(),fmt = '-o',uplims = True, lolims = True,label='0.1%')
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(12, 8))    
#fig1, ax1 = 
ax = ax1.twinx()
ax1.errorbar(wwind_series[0].mean().index,wwind_series[0].mean(),yerr = wwind_series[0].std(),fmt = ':o',uplims = True, lolims = True,label='wind',color = colors[0])
#ax.errorbar(coarse_series[0].mean().index,coarse_series[0].mean(),yerr = coarse_series[0].std(),fmt = ':*',uplims = True, lolims = True,color = colors[1],label ='coarse')
ax.errorbar(clean_coarse_series[0].mean().index,clean_coarse_series[0].mean(),yerr = clean_coarse_series[0].std(),fmt = ':o',uplims = True, lolims = True,color = colors[2],label ='clean_coarse')
ax1.grid()

#ax.grid()
#ax1.set_xlabel("t_value from student t-test",color = colors[2])
#ax.set_xlabel("p_value from student t-test",color = colors[1])
#ax1[1].set_xlim([-2., 2.])
#ax1.#plt.xlable('')
# yy, z_a_agl, 
ax1.set_ylabel("wind speed")
ax.set_ylabel("coarse(#/cc)")
ax1.legend(loc=(0.01, 0.75), fontsize=16)
ax.legend(loc=(0.01, 0.15), fontsize=16)

#plt.errorbar(coarse_series[0].mean().index,coarse_series[0].mean(),yerr = coarse_series[0].std(),fmt = ':o',uplims = True, lolims = True)
#plt.errorbar(cpc_series[2].mean().index,ccn_series[2].mean(),yerr = ccn_series[2].std(),label='0.5%')
#plt.errorbar(cpc_series[3].mean().index,ccn_series[3].mean(),yerr = ccn_series[3].std(),uplims = True, lolims = True,label='0.8%')
#plt.errorbar(cpc_series[4].mean().index,ccn_series[4].mean(),yerr = ccn_series[4].std(),marker = '*',label='1.0%',uplims = True, lolims = True)
#plt.legend()
#plt.ylabel('coarse(#/cc)')
#plt.title('10D_avrg coarse_con')
plt.tight_layout()
fig1.autofmt_xdate()
#%% accumu
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(12, 8))    
#fig1, ax1 = 
ax = ax1.twinx()
accu_series = [df_wind_aero.iloc[clean_small].groupby(Grouper(freq='10D'))['accumulation']]
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['cpc_con']]
#
#plt.errorbar(ccn_series[0].mean().index,ccn_series[0].mean(),yerr = ccn_series[0].std(),fmt = '-o',uplims = True, lolims = True,label='0.1%')
ax.errorbar(accu_series[0].mean().index,accu_series[0].mean(),yerr = accu_series[0].std(),fmt = ':o',uplims = True, lolims = True,label = 'accumulation')
ax1.errorbar(wwind_series[0].mean().index,wwind_series[0].mean(),yerr = wwind_series[0].std(),fmt = ':o',uplims = True, lolims = True,label='wind',color = colors[0])
ax1.grid()
#plt.errorbar(cpc_series[2].mean().index,ccn_series[2].mean(),yerr = ccn_series[2].std(),label='0.5%')
#plt.errorbar(cpc_series[3].mean().index,ccn_series[3].mean(),yerr = ccn_series[3].std(),uplims = True, lolims = True,label='0.8%')
#plt.errorbar(cpc_series[4].mean().index,ccn_series[4].mean(),yerr = ccn_series[4].std(),marker = '*',label='1.0%',uplims = True, lolims = True)
#plt.legend()
ax1.set_ylabel("wind speed")
ax.set_ylabel("accu(#/cc)")
ax1.legend(loc=(0.01, 0.75), fontsize=16)
ax.legend(loc=(0.01, 0.15), fontsize=16)
#plt.ylabel('accumulation(#/cc)')
#plt.title('10D_avrg acc_con')
plt.tight_layout()
fig.autofmt_xdate()
#%%

#%%
wwwind_series = [df_wind_aero.iloc[small_scaler].groupby(Grouper(freq='10D'))['wind_speed']]
wwind_series = [df_wind_aero.iloc[clean_small].groupby(Grouper(freq='10D'))['wind_speed']]
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['cpc_con'],
#                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['cpc_con']]
#
#plt.errorbar(ccn_series[0].mean().index,ccn_series[0].mean(),yerr = ccn_series[0].std(),fmt = '-o',uplims = True, lolims = True,label='0.1%')
plt.errorbar(wwind_series[0].mean().index,wwind_series[0].mean(),yerr = wwind_series[0].std(),fmt = ':o',uplims = True, lolims = True)
plt.errorbar(wwwind_series[0].mean().index,wwwind_series[0].mean(),yerr = wwwind_series[0].std(),fmt = ':o',uplims = True, lolims = True)

plt.ylabel('speed(m/s)')
plt.title('10D_avrg wind')
plt.tight_layout()
fig.autofmt_xdate()
#%%
wind_serires = [df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_1)].groupby(Grouper(freq='M'))['wind_speed'],
                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['wind_speed'],
                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['wind_speed'],
                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['wind_speed'],
                          df_wind_aero.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['wind_speed']]
#
fig = plt.figure()
plt.errorbar(wind_serires[1].mean().index,wind_serires[1].mean(),yerr = wind_serires[1].std(),fmt = ':o',uplims = True, lolims = True,label='0.2%')
#plt.errorbar(wind_serires[2].mean().index,wind_serires[2].mean(),yerr = wind_serires[2].std(),label='0.5%')
#plt.errorbar(wind_serires[3].mean().index,wind_serires[3].mean(),yerr = wind_serires[3].std(),uplims = True, lolims = True,label='0.8%')
#plt.errorbar(wind_serires[4].mean().index,wind_serires[4].mean(),yerr = wind_serires[4].std(),marker = '*',label='1.0%',uplims = True, lolims = True)
#plt.legend()
plt.grid()
plt.ylabel('wind_speed(m/s)')
plt.title('monthly wind')
fig.autofmt_xdate()
#%% seperate error bar
#plt.plot(ccn_series[0].mean()+ccn_series[0].std(),'b.' )
#plt.plot(ccn_series[0].mean()-ccn_series[0].std(),'b.' )
#plt.plot(df_new.iloc[np.intersect1d(small_scaler,ss0_1)].groupby(Grouper(freq='M'))['N_CCN'].mean(),label='0.1%')
#
#
#plt.plot(ccn_series[3].mean()+ccn_series[3].std(),'r.' )
#plt.plot(ccn_series[3].mean()-ccn_series[3].std(),'r.' )
#plt.plot(df_new.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['N_CCN'].mean(),label='.5%')
##plt.plot(df_new.iloc[np.intersect1d(small_scaler,ss0_10)].groupby(Grouper(freq='M'))['N_CCN'].mean(),label='1.0%')
##plt.plot(df_new.iloc[np.intersect1d(small_scaler,ss0_5)].groupby(Grouper(freq='M'))['N_CCN'].mean(),label='0.5%')
##plt.plot(df_new.iloc[np.intersect1d(small_scaler,ss0_2)].groupby(Grouper(freq='M'))['N_CCN'].mean(),label='0.2%')
##plt.plot(df_new.iloc[np.intersect1d(small_scaler,ss0_8)].groupby(Grouper(freq='M'))['N_CCN'].mean(),label='0.8%')
#plt.legend()


#%% merge uhsas and cc&cpc
#uhsas_99 = uhsas_99.set_index('time')
df_aero_ = pd.merge(df_try_10min,uhsas_99,how = 'inner',left_index =True, right_index = True)
#df_aero_[df_aero_.columns[-100:-1]].sum(axis=1)
#%%
#df_ae_drop = df_aero_[df_aero_.columns[-101:-1]]
df_ae_drop = pd.DataFrame()

df_ae_drop['60-300'] = df_aero_[df_aero_.columns[-101:-62]].sum(axis=1)
df_ae_drop['300-500'] = df_aero_[df_aero_.columns[-62:-27]].sum(axis=1)
df_ae_drop['500-700'] = df_aero_[df_aero_.columns[-27:-14]].sum(axis=1)
df_ae_drop['700-1000'] = df_aero_[df_aero_.columns[-14:-1]].sum(axis=1)
df_ae_non_zero = df_ae_drop[(df_ae_drop!=0).any(axis=1)]
df_ae_non_zero['wind_speed'] = df_aero_['wind_speed'][(df_ae_drop!=0).any(axis=1)]
df_ae_non_zero = df_ae_non_zero.dropna()
df_ae_drop['wind_speed'] = df_aero_['wind_speed'][~(df_ae_drop==0).all(axis=1)]
#df_ae_drop = df_ae_drop[clean_boole].dropna() # 2852 rows

df_wibs_drop = df_aero_[['All','wind_speed','FBAP']].dropna()
#df_wibs_drop = df_wibs_drop[(df_wibs_drop!=0).any(axis=1)]
#%% Figure different size~ wind
plt.scatter(df_ae_drop['wind_speed'],df_ae_drop['700-1000'],color = 'red',s = 4,label='.7-1um')
#plt.scatter(df_ae_drop['wind_speed'],df_ae_drop['500-700'],color = 'yellow',s = 4,label='.5-.7um')
#plt.scatter(df_ae_drop['wind_speed'],df_ae_drop['300-500'],color = 'blue',s = 4,label='.3-.5um')
plt.scatter(df_ae_drop['wind_speed'],df_ae_drop['60-300'],color = 'green',s = 4,label='.06-.3um')
plt.scatter(df_aero_['wind_speed'][sss0_2],df_aero_['N_CCN'][sss0_2],s =5,color = 'black',label = '0.2%CCN')
plt.scatter(df_ae_drop['wind_speed'],df_aero_['All'],s= 4,label = '0.5-10um(#/L)',color = 'cyan')
plt.yscale('log')
plt.legend(markerscale = 4)
plt.ylabel('CN(#/cc)')
plt.xlabel('wind speed(m/s)')
plt.ylim((.1,10000))
#%%
plt.plot(np.unique(df_ae_drop['wind_speed']),
         np.poly1d(np.polyfit(df_ae_drop['wind_speed'], np.log(df_ae_drop['700-1000']), 1))(np.unique(df_ae_drop['wind_speed'])),color = 'red')
plt.plot(np.unique(df_ae_drop['wind_speed']),
         np.poly1d(np.polyfit(df_ae_drop['wind_speed'], np.log(df_ae_drop['60-300']), 1))(np.unique(df_ae_drop['wind_speed'])),color = 'green')
plt.plot(np.unique(df_ae_drop['wind_speed']),
         np.poly1d(np.polyfit(df_ae_drop['wind_speed'], np.log(df_ae_drop['300-500']), 1))(np.unique(df_ae_drop['wind_speed'])),color = 'blue')
plt.plot(np.unique(df_ae_drop['wind_speed']),
         np.poly1d(np.polyfit(df_ae_drop['wind_speed'], df_ae_drop['500-700'], 1))(np.unique(df_ae_drop['wind_speed'])),color= 'yellow')


#%$
non_zero_7_10 = np.where(df_ae_non_zero['700-1000']!=0)[0]
non_zero_6_3 = np.where(df_ae_non_zero['60-300']!=0)[0]
non_zero_3_5 = np.where(df_ae_non_zero['300-500']!=0)[0]
non_zero_5_7 = np.where(df_ae_non_zero['500-700']!=0)[0]
#non_zero_7_10 = np.where(df_ae_non_zero['700-1000']!=0)[0]
#%%
plt.plot(np.unique(df_ae_non_zero['wind_speed']),
         np.poly1d(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_6_3], np.log(df_ae_non_zero['60-300'][non_zero_6_3]), 1))(np.unique(df_ae_non_zero['wind_speed'])),color = 'green',label = '0.06-0.3um')
print(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_6_3], np.log(df_ae_non_zero['60-300'][non_zero_6_3]), 1))
plt.plot(np.unique(df_ae_non_zero['wind_speed']),
         np.poly1d(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_3_5], np.log(df_ae_non_zero['300-500'][non_zero_3_5]), 1))(np.unique(df_ae_non_zero['wind_speed'])),color = 'blue',label = '0.3-0.5um')
print(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_3_5], np.log(df_ae_non_zero['300-500'][non_zero_3_5]), 1))
plt.plot(np.unique(df_ae_non_zero['wind_speed']),
         np.poly1d(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_5_7], np.log(df_ae_non_zero['500-700'][non_zero_5_7]), 1))(np.unique(df_ae_non_zero['wind_speed'])),color= 'yellow',label = '0.5-0.7um')
print(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_5_7], np.log(df_ae_non_zero['500-700'][non_zero_5_7]), 1))
plt.plot(np.unique(df_ae_non_zero['wind_speed']),
         np.poly1d(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_7_10], np.log(df_ae_non_zero['700-1000'][non_zero_7_10]), 1))(np.unique(df_ae_non_zero['wind_speed'])),color = 'red',label = '0.7-1.0um')
print(np.polyfit(df_ae_non_zero['wind_speed'][non_zero_7_10], np.log(df_ae_non_zero['700-1000'][non_zero_7_10]), 1))

plt.plot(np.unique(df_wibs_drop['wind_speed']),
         np.poly1d(np.polyfit(df_wibs_drop['wind_speed'], np.log(df_wibs_drop['All']), 1))(np.unique(df_wibs_drop['wind_speed'])),color = 'cyan',label='All')
print(np.polyfit(df_wibs_drop['wind_speed'], np.log(df_wibs_drop['All']), 1))

plt.plot(np.unique(df_wibs_drop['wind_speed']),
         np.poly1d(np.polyfit(df_wibs_drop['wind_speed'], np.log(df_wibs_drop['FBAP']), 1))(np.unique(df_wibs_drop['wind_speed'])),color = 'orange',label = 'FBAP')
print(np.polyfit(df_wibs_drop['wind_speed'], np.log(df_wibs_drop['FBAP']), 1))

plt.ylabel('log_CN(#/cc)')
plt.xlabel('wind speed(m/s)')
plt.legend()
#%%
sss0_2 = np.logical_and(df_aero_['SS']==0.2,clean_boole)
plt.scatter(df_ss0_2['accumulation'],df_ss0_2['N_CCN'])
#plt.yscale('log')
#plt.xscale('log')
df_ss0_2 = df_aero_[np.logical_and(df_aero_['new_flag']==True , df_10min['SS']==0.2)][['N_CCN','accumulation']].dropna()
df_ss0_2['wind_speed'] = df_aero_['wind_speed'][np.logical_and(df_aero_['new_flag']==True , df_10min['SS']==0.2)]
df_ss0_2['wind_flag'] = dff_uhsas['wind_flag'][np.logical_and(df_aero_['new_flag']==True , df_10min['SS']==0.2)]
#%%
fig = plt.figure(figsize = (6,4))
plt.scatter(df_ss0_2['accumulation'][df_ss0_2['wind_flag']==0],df_ss0_2['N_CCN'][df_ss0_2['wind_flag']==0],color = 'purple',s=4, label = '0-6m/s')
plt.scatter(df_ss0_2['accumulation'][df_ss0_2['wind_flag']==1],df_ss0_2['N_CCN'][df_ss0_2['wind_flag']==1],color = 'orange',s=4, label = '6-12m/s')
plt.scatter(df_ss0_2['accumulation'][df_ss0_2['wind_flag']==2],df_ss0_2['N_CCN'][df_ss0_2['wind_flag']==2],color = 'blue',s=4, label = '12-18m/s')
plt.scatter(df_ss0_2['accumulation'][df_ss0_2['wind_flag']==3],df_ss0_2['N_CCN'][df_ss0_2['wind_flag']==3],color = 'green',s=4, label = '18-24m/s')
plt.legend(markerscale = 4,fontsize = 14)
plt.ylabel("N_CCN(#/cc)")
plt.xlabel("accumulation mode(#/cc)")
#%%
ax = plt.gca()
ax.scatter(df_aero_['lat'][clean_boole] ,df_ae_drop['60-300'][clean_boole],label = '60-300nm',color = 'green')
axs = ax.twinx()
axs.scatter(df_aero_['lat'][clean_boole] ,df_aero_['FBAP'][clean_boole],label = 'FBAP',color='red')
axs.set_yscale('log')
plt.xlabel('lat')
axs.set_ylabel('FBAP(0.5-10um)')
ax.set_ylabel('CN(60-300) #/cc')
#%%
#plt.scatter(df_aero_['lat'][clean_boole] ,df_ae_drop['300-500'][clean_boole])
#plt.scatter(df_aero_['lat'][clean_boole] ,df_ae_drop['500-700'][clean_boole])
ax = plt.gca()
ax.scatter(df_aero_['lat'][clean_boole] ,df_ae_drop['60-300'][clean_boole],label = '60-300nm',color = 'green')
axs = ax.twinx()
axs.scatter(df_aero_['lat'][clean_boole] ,df_aero_['rain'][clean_boole],label = 'FBAP',color='red')
axs.set_yscale('log')
ax.set_yscale('log')
ax.set_xlabel('lat')
axs.set_ylabel('rain intensity(mm/h)')
ax.set_ylabel('CN(60-300) #/cc')
axs.set_ylim((0.1,25))
ax.set_ylim((0.01,100000))
#%%
ax = plt.gca()
ax.scatter(df_aero_['lat'][clean_boole] ,df_ae_drop['60-300'][clean_boole],label = '60-300nm',s = 5,color = 'green')
axs = ax.twinx()
axs.scatter(df_aero_['lat'][clean_boole] ,df_aero_['wind_speed'][clean_boole],label = 'FBAP',s=5,color='red')
axs.set_yscale('log')
plt.xlabel('lat')
axs.set_ylabel('wind speed(m/s)')
ax.set_ylabel('CN(60-300) #/cc')
#%%
ax = plt.gca()
ax.scatter(df_aero_['wind_direction'][clean_boole] ,df_ae_drop['60-300'][clean_boole],label = '60-300nm',s = 5,color = 'green')
#axs = ax.twinx()
#axs.scatter(df_aero_['lat'][clean_boole] ,df_aero_['wind_direction'][clean_boole],label = 'FBAP',s=5,color='red')
ax.set_yscale('log')
plt.xlabel('direction')
ax.set_ylim((0.01,100000))
#axs.set_ylabel('wind direction(degree)')
ax.set_ylabel('CN(60-300) #/cc')
#%%
ax = plt.gca()
ax.scatter(df_aero_['rain'][clean_boole] ,df_ae_drop['60-300'][clean_boole],label = '60-300nm',s = 5,color = 'green')
#axs = ax.twinx()
#axs.scatter(df_aero_['lat'][clean_boole] ,df_aero_['wind_direction'][clean_boole],label = 'FBAP',s=5,color='red')
ax.set_yscale('log')
plt.xlabel('rain intensity')
ax.set_ylim((0.01,100000))
ax.set_xlim((0.01,20))
#axs.set_ylabel('wind direction(degree)')
ax.set_ylabel('CN(60-300) #/cc')