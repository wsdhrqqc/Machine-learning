#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:32:21 2019

@author: qingn
"""


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
import mpl_toolkits
#import mpl_toolkits.basemap as bm
#from mpl_toolkits.basemap import Basemap, cm
import act
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
var = 'concentration'
# %%
#path_exhaust_id = '/scratch/wsdhr/marcus_data/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc'
#path_cpc = '/scratch/wsdhr/marcus_data/maraoscpcfM1.b1_1029-0324/maraos*.nc'
#cpc = arm.read_netcdf(path_cpc)
#exhaust_id = netCDF4.Dataset(path_exhaust_id)
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')