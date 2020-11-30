# -*- coding: utf-8 -*-
"""
Spyder Editor
Figures of marcus cloud mask
This is a temporary script file. What is am trying to do isc
"""

import xarray as xr
import dask
import numpy as np
import numpy.ma as ma
import matplotlib.backends.backend_pdf
import scipy.stats as stats
import act.io.armfiles as arm
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
import pyart
matplotlib.use('Agg')
import sys
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

#import pathlib
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 12
LABELSIZE = 12
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}
#%%
def arm_read_netcdf(directory_filebase,time_resolution):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>
    : time_resolution needs to be a string'''
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time=time_resolution).nearest()
    return file

def arm_read_netcdf_ori(directory_filebase):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>
    : time_resolution needs to be a string'''
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
#    file = file_ori.resample(time=time_resolution).nearest()
    return file_ori
#%%
    
home = os.path.expanduser("/condo/mcfarq/wsdhr")
# main save directory

thesis_path = os.path.join(home,'full_WACR')
# piclke file data (under NQ)
#pkl = os.path.join(os.getcwd(), "pkl")
#pkl = os.path.join(home, "pkl")
#/Users/qingn/Desktop/NQ/personal/thesis/IMG_5047.jpg

# figure save directory
figpath = os.path.join(home, "Figure")
if not os.path.exists(figpath):
    os.mkdir(figpath)
# close all figures
plt.close("all")
#%% Read in data
files = glob.glob('/condo/mcfarq/wsdhr/full_WACR/mar*.nc')
files = sorted(files)
cmap = 'pyart_NWSRef'

def plot_wacr_reflectivity(i):
    plt.close("all")
    ''' No.i files in wacr
    '''
    wacr = arm.read_netcdf(files[i])
    
    #%% Plotting
    fig = plt.figure(figsize = (12,3))
    ax = plt.gca()
    
    
    plt.contourf(wacr.time.values, wacr.height.values,wacr.cloud_mask_95ghz_kollias.T)#,label = 'Micro-pulse lidar cloud mask')
    ec=plt.contourf(wacr.time.values, wacr.height.values,wacr['reflectivity_best_estimate'].T,cmap = 'jet')#,label = 'Hydrometeor-only 95GHz')
    
    
    plt.plot(wacr.time.values,wacr.cloud_base_best_estimate,'w.',alpha = 0.3,label = 'cloud_base',markersize=3)
    
    fig.autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Height(m)')
    plt.yscale('log')
    plt.legend()
    #plt.show()
    
    cbar = plt.colorbar(ec)
    cbar.set_label('reflectivity(dBZ)')
    #%%
    
    fig.tight_layout()
    fsave = "Full_cloud_mask_"+str(wacr.reflectivity.time[0].values)[:13]
    fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
    
for i in range(127):
    plot_wacr_reflectivity(i)
    #%%

    
    
