#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:11:35 2019
Poster2
@author: qingn
"""
from datetime import datetime, timedelta
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
#import datetime
import matplotlib
import sys
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
import mpl_toolkits
#import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
import act
import module_ml
from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}
FIGWIDTH = 10
FIGHEIGHT = 6
FONTSIZE = 16
datastream = 'maraoscpc'
var = 'concentration'
qc_var = 'qc_concentration'
# %% load data and metadata
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201711*'
path_co = '/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.201711*'
cpc = arm.read_netcdf(path_cpc)
co = arm.read_netcdf(path_co)
unit_cpc = cpc[var].units
feature_cpc = cpc[var].long_name
unit_co = co['co'].units
feature_co = co['co'].long_name
# %% pull up variables
df_cpc = cpc.to_dataframe()
df_co = co.to_dataframe()
sec_res_cpc=df_cpc.concentration.resample('S').mean()
sec_res_co=df_co.co.resample('S').mean()

sec_res_qccpc=df_cpc.qc_concentration.resample('S').mean()
sec_res_qcco=df_co.qc_co.resample('S').mean()

cpc_con_ms = np.ma.masked_where(sec_res_qccpc>0,sec_res_cpc)
co_con_ms = np.ma.masked_where(sec_res_qcco>0,sec_res_co)
t_normal = np.arange(datetime(2017,11,1,0,0,0), datetime(2017,12,1,0,0,0), timedelta(seconds=1)).astype('datetime64[ns]')