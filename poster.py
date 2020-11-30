#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:15:29 2019
Poster for AI symposium 
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
cpc_con = df_cpc[var].values
co_con = df_co['co'].values
qc_cpc_con = df_cpc[qc_var].values
qc_co = df_co['qc_co'].values
time_cpc = cpc['time'].values
time_co = co['time'].values
# %% histgram
plt.figure(figsize = (FIGWIDTH*1.2,FIGHEIGHT))
plt.subplot(211)
plt.hist(cpc_con)
plt.title('hitogram_for_%s'%(feature_cpc),fontsize = FONTSIZE)
plt.xlabel("concentration "+ unit, fontsize = FONTSIZE)
plt.ylabel("Counts", fontsize = FONTSIZE)
#plt.ylim((0,69999))

plt.subplot(212)
plt.hist(cpc_con)
plt.title('hitogram_for_%s'%(feature_cpc),fontsize = FONTSIZE)
plt.xlabel("concentration "+ unit, fontsize = FONTSIZE)
plt.ylabel("Counts", fontsize = FONTSIZE)
plt.xlim((0,50000))
plt.tight_layout()
fig.autofmt_xdate()

plt.figure(figsize = (FIGWIDTH*1.2,FIGHEIGHT))
plt.subplot(211)
plt.hist(co_con)
plt.title('hitogram_for_%s'%(feature_co),fontsize = FONTSIZE)
plt.xlabel("concentration "+ unit, fontsize = FONTSIZE)
plt.ylabel("Counts", fontsize = FONTSIZE)
plt.subplot(212)
plt.hist(co_con, 3000, density=True)
plt.title('hitogram_for_%s'%(feature_co),fontsize = FONTSIZE)
plt.xlabel("concentration "+ unit, fontsize = FONTSIZE)
plt.ylabel("Counts", fontsize = FONTSIZE)
plt.xlim((0,5))
plt.tight_layout()
fig.autofmt_xdate()
# %%
arm_bad_ratio_cpc = np.size(np.where(qc_cpc_con>0))/np.size(cpc_con)
arm_bad_ratio_co = np.size(np.where(qc_co>0))/np.size(co_con)
print("ARM cpc qc>0 ratio is %s"%(arm_bad_ratio_cpc))
print("ARM co qc>0 ratio is %s"%(arm_bad_ratio_co))
# %% Visualization
print(df.describe())
print(df.columns)
print(pd.isna(df[var]).any())
# %%

N = 86400

fig = plt.figure(figsize = (FIGWIDTH*1.2,FIGHEIGHT))
plt.hist(qc_cpc_con)
plt.ylim((0,1000))
plt.xlim((0,962))
#plt.plot(time[N*21+30000:(N*21+43200)], qc_cpc_con[N*21+30000:(N*21+43200)])
plt.tight_layout()
fig.autofmt_xdate()
# %% 


# %% Pre-process
'''first off, we mask the values when instrument is not working well'''
a_con = np.ma.masked_where(qc_cpc_con>0,cpc_con)
b_con = np.ma.masked_where(qc_co>0,co_con)
# %%
t = np.arange(datetime(2017,11,1,0,0,0), datetime(2017,12,1,0,0,0), timedelta(seconds=1)).astype('datetime64[ns]')
#start_int1 = np.datetime64("2017-11-01 00:00:00", 'ns')
#end_int1 = np.datetime64("2017-11-30 00:00:00",'ns')
#time_normal1 = (range(start_int1, end_int1))
##
#start_int = int(np.datetime64("2017-11-01 00:00:00", "ns").astype(datetime.datetime)/10**9)
#end_int = int(np.datetime64("2017-11-30 00:00:00", "ns").astype(datetime.datetime)/10**9)
#time_normal = np.array(range(start_int, end_int))
# %%
c= '2017-11-11T01:38:07.350000000'
    stime=['010738','011133']
    etime=['011017','011220']
fig = plt.figure(figsize = (FIGWIDTH*1.8,FIGHEIGHT))
#plt.hist(time)
plt.plot(time[N*9+78000-300:N*9+78120],a_con[N*9+78000-300:N*9+78120])
plt.tight_layout()
fig.autofmt_xdate()
# %%
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

#%%
def getAOSData(files):
    period = 15
    minp = 5

#   obj = arm.read_netcdf(files,variables=var)
    obj = arm.read_netcdf(files)
    diff = obj[var].diff('time', n=1)
    time = obj['time'].values

    ds = diff.to_dataset(name='cpc_diff')
    ds['cpc_diff'].values = abs(diff.values)
#    ds = ds.rolling(time=period, min_periods=minp,center=True).mean()
    #ds['cpc'] = obj[var]

    return ds, obj[var]
# %%
def machine_learning(file_path,files_path):
    #Use ADC example script to get the data
    # ''.join(['./',datastream,'/*',sdate,'*'])
    sdate = '20171111'
    edate = sdate
    files = glob.glob(file_path)
    ds,cpc = getAOSData(files)
    time = ds['time'].values

    stime=['013807','011133']
    etime=['011017','011220']
    sbad=[]
    ebad=[]
    
    for i in range(len(stime)):
        sdummy=np.datetime64(''.join(['-'.join([sdate[0:4],sdate[4:6],sdate[6:8]]),
            'T',':'.join([stime[i][0:2],stime[i][2:4],stime[i][4:6]])]))
        sbad.append(sdummy)
        edummy=np.datetime64(''.join(['-'.join([edate[0:4],edate[4:6],edate[6:8]]),
            'T',':'.join([etime[i][0:2],etime[i][2:4],etime[i][4:6]])]))
        ebad.append(edummy)

    #Create the flag that we want to train for.. I.e. exhaust, no exhaust
    y_train=np.zeros(len(time))
    all_indices=[]
    for i in range(len(sbad)):
        idx=(time >= sbad[i])*(time <= ebad[i])
        all_indices.append(list(np.where(idx)[0]))

    #Set indices of previous periods to bad
    y_train[all_indices[0]]=1.

    #Sets different values for the RandomForestClassifier
    md=5 #Max depth of branches
    nest=25  #Total number of trees in the forest
    leafs=5 # Min number of leafs to use

    #Setup the model using 16 cores
    #Random_state=0 gaurantees that we will get the same result each time
    model = RandomForestClassifier(n_estimators = nest,max_depth=md,random_state=0,min_samples_leaf=leafs,n_jobs=16)

    #Fit the model  to the training dataset
    model.fit(ds.to_dataframe(),y_train)

    #If using a RandomForest, print out the feautre importance values
    #This will tell us what weight each variable had on the result
    try:
        fi=model.feature_importances_
        col=ds.columns
        print(col)
        print(fi)
        type='RandomForest'
    except:
        type='KNeighbors'
        
        
    #Get Data to apply the ML algorith to
    args = sys.argv
    if len(args) > 1:
        sdate = str(args[1])
        edate = sdate
    if len(args) > 2:
        edate = str(args[2])
    sdate = '20171029'
    edate = '20180324'
    
#    ''.join(['./',datastream,'/mar*.nc'])
    files = glob.glob(files_path)
    files.sort()
    files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]

    obs_ds,cpc = getAOSData(files)
    
    X_test = obs_ds.to_dataframe()

    # 2 methods that could be used
    #This one just goes off the model prediction of 0 or 1 
#    result = model.predict(X_test)
#    idx = (result == 0)
#    idx = np.append(idx,True)
    '''parallel'''
#    Working method uses the probabilites that the model ouputs that a point is exhaust
#    Smooths that data out so we don't get noisy flagging every other time
    prob=model.predict_proba(X_test)[: ,1]
    prob=pd.DataFrame(data=prob).rolling(min_periods=5 ,window=60*10 ,center=True).mean().values.flatten()
#
#    Flags anything with a probability higher than 1.5% which seems very small
#    but actually works out very well
    fflag=0.00001
#    fflag=0.9999999
    idx=(prob >= fflag)
    idx = np.append(idx,True)
    index=np.where(idx)

#    E = int(1.5*N)
#    S = int(0.5*N)
    #The rest of the program is plotting up the data for visualization and testing purposes
    time=X_test.index.to_pydatetime()
    return index