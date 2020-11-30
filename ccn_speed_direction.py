#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 23:32:01 2019
to see the CCN distribution
@author: qingn
"""

import xarray as xr
import dask
import numpy as np
import matplotlib as mpl
#import astral
import pandas as pd
import netCDF4
import os
from act.io.armfiles import read_netcdf
import datetime
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
import mpl_toolkits
#import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
import act

import act.io.armfiles as arm
import act.plotting.plot as armplot
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
FIGURESIZE = (14,6)
FONTSIZE = 14
# %%
#ccn = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccn/maraosccn*.nc')
ccn_colavgfeb = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccnfeb/maraosccn1colavgM1.b1*.nc')
#ccn_colavg = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1*.nc')
#ccn100 = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosccn100M1/maraosccn100M1.a1*.nc')
ccncolspectra = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosccn1colspectraM1/maraosccn1colspectraM1.b1*.nc')
ccncol = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosccn1colM1/maraosccn1colM1.b1*.nc')
# met = arm.read_netcdf('/Users/qingn/Documents/ARM/ACT/maraosmetM1_1.a1/maraosmetM1.a1.20171031.000000.nc')
# cloud condensation nuclei particle counter: 
# measures the concentration of aerosol particles by drawing
# an air sample through a column with thermodynamically unstable supersaturated
# water vapor that can conndense onto aerosol particles. Counted and sized by 
# an optical particle counter (OPC), in this way the CCN measures activated
# ambient aerosol particle number concetration as a function of supersaturation
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1.20180205.000222.nc')
# %%

plt.figure(figsize=[FIGURESIZE[0]*1.5,FIGURESIZE[1]])
plt.plot(ccncol['time'][:43200],ccncol['N_CCN'][:43200])
plt.title('maraosccn1colM1.b1(per s)_CCN_concentration',fontsize=26)
plt.tight_layout()


# %%
plt.figure(figsize=[FIGURESIZE[0]*1.5,FIGURESIZE[1]])
plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,1], label = '0.1%')
plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,2], label = '0.2%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,2])|
plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,3], label = '0.5%')
plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,4], label = '0.8%')
plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,5], label = '1.0%')
plt.legend('%s%%'%([ccn_colavgfeb['setpoint'].values[i] for i in range(6)]))
#plt.legend({'%s%%'%()}
plt.title('maraosccnspectraM1.b1(per h)_CCN_concentration',fontsize=26)
plt.legend(fontsize = FONTSIZE)
plt.yscale('log')
plt.tight_layout()
# %%
plt.figure(figsize=[FIGURESIZE[0]*1.5,FIGURESIZE[1]])


plt.plot(ccn_colavgfeb['time'][1:120],ccn_colavgfeb['N_CCN'][1:120], label = '0.1%')
plt.plot(ccncolspectra['time'][:24],ccncolspectra['N_CCN'][:,1][:24], label = '0.1%')
plt.plot(ccncolspectra['time'][:24],ccncolspectra['N_CCN'][:,2][:24], label = '0.2%')
plt.plot(ccncolspectra['time'][:24],ccncolspectra['N_CCN'][:,3][:24], label = '0.5%')
plt.plot(ccncolspectra['time'][:24],ccncolspectra['N_CCN'][:,4][:24], label = '0.8%')
plt.plot(ccncolspectra['time'][:24],ccncolspectra['N_CCN'][:,5][:24], label = '1.0%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,3])
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,2], label = '0.2%')
##plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,2])|
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,3], label = '0.5%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,4], label = '0.8%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,5], label = '1.0%')
plt.legend(fontsize = FONTSIZE)
#plt.legend('%s%%'%([ccn_colavgfeb['setpoint'].values[i] for i in range(6)]))
#plt.legend({'%s%%'%()}
plt.title('maraosccn1colavgM1.b1(per ~12min)_CCN_concentration',fontsize=26)
#plt.legend(fontsize = FONTSIZE)
plt.yscale('log')
plt.tight_layout()

#%%
plt.figure(figsize=[FIGURESIZE[0]*0.8,FIGURESIZE[1]])
distribution = ccn_colavgfeb['N_CCN_dN']/ccn_colavgfeb['Q_sample']
#plt.plot(ccn_colavgfeb['N_CCN_dN'][55,:])
plt.plot(distribution[90,:]*ccn_colavgfeb['droplet_size']*np.log(10))
#plt.plot(distribution[55,:])
#plt.plot(ccn_colavgfeb['time'][1:120],ccn_colavgfeb['N_CCN'][1:120], label = '0.1%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,2], label = '0.2%')
##plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,2])|
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,3], label = '0.5%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,4], label = '0.8%')
#plt.plot(ccncolspectra['time'],ccncolspectra['N_CCN'][:,5], label = '1.0%')
#plt.legend('%s%%'%([ccn_colavgfeb['setpoint'].values[i] for i in range(6)]))
#plt.legend({'%s%%'%()}
plt.xlabel('bin_size(um_log_scale)',fontsize =17)
#plt.xlabel('xlabel', fontsize=13)
plt.ylabel('dN/dlogDp',fontsize = 18)
plt.title('maraosccn_colavgM1.b1_n_time(t)',fontsize=26)
#plt.legend(fontsize = FONTSIZE)
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()

# %%
plt.figure(figsize=[FIGURESIZE[0]*0.8,FIGURESIZE[1]])
plt.plot(ccn_colavgfeb['N_CCN_dN'][55,:])
#plt.plot(distribution[90,:]*ccn_colavgfeb['droplet_size']*np.log(10))
plt.xlabel('bin_size(um)',fontsize =17)
#plt.xlabel('xlabel', fontsize=13)
plt.ylabel('counts/second in each bin',fontsize = 18)
plt.title('maraosccn_colavgM1.b1(per sec)_N_CCN_dN_time(t)',fontsize=26)
#plt.legend(fontsize = FONTSIZE)
#plt.yscale('log')
#plt.xscale('log')
plt.tight_layout()
# %% read in all variables

droplet_size = ccn['droplet_size'].values
droplet_size_bounds = ccn['droplet_size_bounds'].values
'''Volumetric flow rate of sample air cm^3/min'''
Q_sample = ccn['Q_sample'].values
'''Number of particles that are larger than 10um'''
anc = ccn['aerosol_number_concentration'].values 
supersaturation = ccn['supersaturation_calculated'].values 
n_ccn = ccn['N_CCN'].values 
qc_N_CCN = ccn['qc_N_CCN'],values
n_ccn_dn = ccn['N_CCN_dN'].values
overflow = ccn['overflow'].values
#setpoint = ccn['setpoint'].values all zeroes

print(setpoint)
#first_bin = ccn['first_bin_used'].values all zeroes
time = ccn['time'].values
time_offset = ccn['time_offset'].values
time_bounds = ccn['time_bounds'].values
'''variables(dimensions): int32 base_time(), float64 time_offset(time), float64 time(time), float64 time_bounds(time,bound), float32 droplet_size(droplet_size), float32 droplet_size_bounds(droplet_size,bound), float32 setpoint(setpoint), float32 Q_sample(time), float32 overflow(time), float32 supersaturation_calculated(time), float32 aerosol_number_concentration(time), float32 N_CCN(time), int32 qc_N_CCN(time), float32 N_CCN_dN(time,droplet_size), int32 first_bin_used(time), float32 lat(), float32 lon(), float32 alt()'''
# %%
ws = met['wind_speed'].values
wd = met['wind_direction'].values

