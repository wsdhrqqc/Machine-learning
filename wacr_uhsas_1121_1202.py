#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:26:02 2020

@author: qingn
"""

import numpy as np 
import matplotlib.backends.backend_pdf
#import matplotlib as mpl
#import astral
import pandas as pd
import glob
import netCDF4
import os
import matplotlib.pyplot as plt
import module_ml
import act.io.armfiles as arm
import matplotlib.transforms as mtransforms
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 8
LABELSIZE = 8
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


matplotlib.rc('xtick', labelsize=LABELSIZE) 
matplotlib.rc('ytick', labelsize=LABELSIZE)
 # home directory
home = os.path.expanduser("/Users/qingn/Desktop/NQ")
# main save directory

thesis_path = os.path.join(home,'personal','thesis')
# piclke file data (under NQ)
#pkl = os.path.join(os.getcwd(), "pkl")
#pkl = os.path.join(home, "pkl")
#/Users/qingn/Desktop/NQ/personal/thesis/IMG_5047.jpg

# figure save directory
figpath = os.path.join(thesis_path, "Figures")
if not os.path.exists(figpath):
    os.mkdir(figpath)
# close all figures
plt.close("all")
#%
# %%  Hobart to Davis: 1121-1202
cpc_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2*'))
uhsas_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1*'))
co_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.20*'))
wacr_files = sorted(glob.glob('/Volumes/Extreme SSD/WACR/mararsclwacr1kolliasshpM1.c1*.nc'))
path_o3 = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraoso3/maraoso3M1.b1.201*.custom.nc'))
nav_dir = sorted(glob.glob('/Users/qingn/Desktop/NQ/marnav/marnavbeM1.s1.201*'))
met_dir = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201*.nc'))
sonde_dir = sorted(glob.glob('/Users/qingn/Desktop/NQ/sounde/marsondewnpnM1.b1.201*'))


wacr = arm.read_netcdf(wacr_files[21:32])
cpc = arm.read_netcdf(cpc_files[24:36])
uhsas = arm.read_netcdf(uhsas_files[23:35])
co = arm.read_netcdf(co_files[32:44])
o3 = arm.read_netcdf(path_o3[23:35])
nav = arm.read_netcdf(nav_dir[23:35])
wind = arm.read_netcdf(met_dir[47:59])
sonde = arm.read_netcdf(sonde_dir[81:125])


#%% wacr & cpc & uhsas
refl = wacr['reflectivity_best_estimate'].resample(time = '1min').nearest()
base = wacr['cloud_base_best_estimate'].resample(time = '1min').nearest()
miss = wacr['missing_data_flag'].resample(time = '1min').nearest()

_, index1 = np.unique(cpc['time'], return_index = True)
cpc = cpc.isel(time = index1)
qc_con = cpc['qc_concentration']
cpc_con = cpc['concentration']
print(np.unique(qc_con))
#%

cpc_con = cpc_con.where((qc_con!=2)&(qc_con!=65474)&(qc_con!=962))
cpc_con_10 = cpc_con.resample(time = '1min').mean().resample(time = '10min').nearest()
cpc_con_10s = cpc_con.resample(time='10s').nearest()
lower = uhsas['lower_size_limit'].values[0]

_, index1 = np.unique(uhsas['time'], return_index = True)
uhsas = uhsas.isel(time = index1)
uhsas = uhsas.resample(time='10s').nearest(tolerance = '10s')
uhsas_con = uhsas['concentration'][:,4:].sum(dim='bin_num') # starting from 67.22-1000nm

sog = nav.speed_over_ground[::100]
en = nav.en_route[::100]
surge = nav.surge_velocity[::100]
a = nav.surge_acceleration[::100]
#%%
#%% See how many uhsas have been removed
idx = np.where(cpc_con_10s.values-uhsas_con.values<0)[0]
plt.plot(uhsas_con.time,uhsas_con.values,'g.')
plt.plot(cpc_con_10s.time,cpc_con_10s.values,'b.')
# plt.plot(cpc_con_10s[1440:].values-uhsas_con.values)

plt.plot(uhsas_con[idx].time,uhsas_con[idx].values,'r.',markersize=5)
plt.plot(cpc_con_10s[idx].time,cpc_con_10s[idx].values,'k.',markersize=5)


plt.ylim(0,2000)
print(sum(cpc_con_10s.values-uhsas_con.values<0))
# np.where()
# looks like abnormal uhsas happens when contamination happen, therefore i am ignoring it right now
#%% uhsas qc
# uhsas_con = uhsas_con.where(cpc_con_10s.values-uhsas_con.values>0)
uhsas_con_10 = uhsas_con.resample(time='1min').mean().resample(time = '10min').nearest(tolerance='10min')
con1 = uhsas['concentration'][np.where(cpc_con_10s.values-uhsas_con.values>0)[0],:]
# mask abnormal uhsas
# mask_land = 1 * np.ones((ds.dims['latitude'], ds.dims['longitude'])) * np.isnan(ds.sst.isel(time=0))  
mask = np.zeros([len(uhsas_con),99])
mask[cpc_con_10s.values-uhsas_con.values>0,:]=1
# mask_array = mask_ocean + mask_land
con = uhsas['concentration'][:,4:].where(mask[:,4:])
# .resample(time='1min').mean().resample(time = '10min').nearest(tolerance='10min')
# con = uhsas_con.resample(time='10min').nearest()
#%% co & o3
_, index1 = np.unique(co['time'], return_index = True)
co = co.isel(time = index1)
co_con = co['co_dry'].where(co['qc_co_dry']<16384)
co_con_10 = co_con.resample(time = '1min').mean().resample(time = '10min').nearest()

_, index1 = np.unique(o3['time'], return_index = True)
o3 = o3.isel(time = index1)
o3_con=o3['o3'].where((o3['qc_o3']<262144))
o3_con_10 = o3_con.resample(time='1min').mean().resample(time = '10min').nearest()
#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
#%% id
cpc_full = arm.read_netcdf(cpc_files)
datastream = 'maraoscpc'
var = 'concentration'
index_cpc_ml= module_ml.machine_learning(''.join(['./',datastream,'/*','20171111','*']) ,''.join(['./',datastream,'/mar*.nc'])) 
index_cpc_ml = np.array(index_cpc_ml[0])
flg = np.zeros(len(cpc_full['concentration']))
flg[index_cpc_ml] = 1
cpc_full['flag'] = flg


idx1 = find_closest(cpc_full.time.values, cpc_con_10.time.values)
flag_ml = cpc_full.flag[idx1]
index_ml = np.array(np.where(flag_ml==1)[0])
#%%
plt.plot(sonde.wspd[(sonde.alt>30)&(sonde.alt<50)],'k.')
#%% combine and share x
fig = plt.figure(figsize = (12,12))

ax = plt.subplot(611)

ax.plot(wind.wind_direction[::30].time,wind.wind_direction[::30],'k.',label='rel_dir',markersize=2)
# ax.plot(wind.wind_direction[::30][wind.wind_direction].time,wind.wind_direction[::30],'b.',label='rel_dir',markersize=2)
ax.set_ylabel('degree')
ax.plot(sonde.deg[(sonde.alt>30)&(sonde.alt<50)].time,sonde.deg[(sonde.alt>30)&(sonde.alt<50)],'y*',label='sonde_deg')
ax.legend()
ax1 = ax.twinx()
ax1.plot(wind.wind_speed[::30].time,wind.wind_speed[::30],'b.',label='rel_sp',markersize=2,alpha=0.5)
# plt.plot(wind.wind_direction[::30].time,wind.wind_direction[::30],label='rel_dir(deg)')
ax1.plot(sonde.wspd[(sonde.alt>30)&(sonde.alt<50)].time,sonde.wspd[(sonde.alt>30)&(sonde.alt<50)],'r*',label='sonde_wspd')
ax1.set_ylabel('m/s',color='blue')
ax1.legend()

ax2 = plt.subplot(612,sharex=ax)
ax2.plot(surge.time,surge.values,'k.',label ='heading v',markersize=3)
ax2.plot(a.time,a-4,'b.',label ='heading a-4',markersize=3)
ax2.plot(sog.time,sog-surge,'g.',label ='sog-sur(m/s)',markersize=3)
ax2.plot(en[en==0].time,en[en==0],'r.',color='yellow''',label ='0:port')
ax2.legend(fontsize=6)
ax2.set_ylabel('m/s')

ax3 = plt.subplot(613,sharex=ax)
ax3.plot(co_con_10.time,co_con_10.values,'k.')
ax3.plot(co_con_10[index_ml].time.values,co_con_10[index_ml].values,'r.')
a1 = ax3.twinx()
a1.plot(o3_con_10.time,o3_con_10.values,'g.')
a1.plot(o3_con_10[index_ml].time.values,o3_con_10[index_ml].values,'b.')
ax3.set_ylabel('co(ppmv)',color = 'red')
a1.set_ylabel('o3(ppbv)',color = 'green')
ax3.set_yscale('log')

ax4 = plt.subplot(614,sharex=ax)
ax4.plot(cpc_con_10.time,cpc_con_10.values,'k.')
ax4.plot(cpc_con_10[index_ml].time.values,cpc_con_10[index_ml].values,'r.',label = 'cpc')
ax4.plot(uhsas_con_10.time,uhsas_con_10.values,'g.')
ax4.plot(uhsas_con_10[index_ml].time.values,uhsas_con_10[index_ml].values*0.1,'b.',label = 'uhsas(67-1000)*0.1')
# ax.plot(uhsas_con_10[index_ml].time.values,uhsas_con_10[index_ml].values*0.1,'b.',label = 'uhsas(67-1000)*0.1')
# ax4.set_yscale('log')
ax4.set_ylim((10,2000))
ax4.set_ylabel('CN(#/cc)')
ax4.legend(fontsize=8)

ax5 = plt.subplot(615,sharex=ax)
ec = ax5.pcolormesh(con.time.values,lower[4:],np.log(con).T)
ax5.set_ylabel('bin(nm)')
# plt.colorbar(ec)

ax6 = plt.subplot(616,sharex=ax)
ea = ax6.pcolormesh(refl[::20].time,refl.height,refl[::20].T,vmin=-15,vmax=20,cmap='Set1_r')
# plt.colorbar(ea,orientation='horizontal')
ax6.plot(base[::20].time.values,base[::20].values,'k.',alpha = 0.3,label = 'cloud base')
ax66 = ax6.twinx()
ax66.plot(wind.rain_intensity[wind.rain_intensity>0.1][::40].time.values,wind.rain_intensity[wind.rain_intensity>0.1][::40],
          'g.',label = 'rain_int',markersize=3)
trans = mtransforms.blended_transform_factory(ax6.transData, ax6.transAxes)
ax6.fill_between(miss[::40].time.values, 0, 1000,
                 where= (miss[::40].values==1)|(miss[::40].values==4)|(miss[::40].values==5)|(miss[::40].values==7),
                facecolor='orange', alpha=0.1, transform=trans,label='Radar missing')
ax66.legend()
ax66.set_ylabel('mm/h',color = 'green')
# ax.fill_between(miss[::40].time.values, 0, 1000, where = miss[::40]!=0 ,
#                 facecolor='orange', alpha=0.1, transform=trans,label='Radar missing')

# plt.legend()
ax6.set_yscale('log')
ax6.set_ylabel('height(m)')

fig.autofmt_xdate()
