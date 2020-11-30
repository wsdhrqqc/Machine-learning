


# %%
import xarray as xr
import dask
import numpy as np
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
from mpl_toolkits.basemap import Basemap, cm
import act
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier
matplotlib.rc('xtick', labelsize=26) 
matplotlib.rc('ytick', labelsize=26) 
# %%
cpc = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraos*.nc')
#exhaust_id = arm.read_netcdf('/Users/qingn/Desktop/NQ/exhaust_id/AAS*.nc')
co = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosco/mar*.nc')
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')

time_cpc = cpc['time'].values
time_id = exhaust_id['time'].values
time_co = co['time'].values

#%%
