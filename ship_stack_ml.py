#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:49:23 2020
Machine learning is great in identification for MARCUS
@author: qingn
"""
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib.dates import DateFormatter,date2num

import act.io.armfiles as arm
#import act.plotting.plot as armplot
#import act.discovery.get_files as get_data

import glob
import matplotlib.pyplot as plt
import os
import json
import matplotlib
import sys
import numpy as np
import xarray as xr
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
#%% parameter 
period = 15
minp = 5
#Sets different values for the RandomForestClassifier
md=5 #Max depth of branches
nest=25  #Total number of trees in the forest
leafs=5 # Min number of leafs to use
# logics here: run Adam's algorithms first, combine co and cpc results together
# to have a summary, 先完成后完美
#%% both cpc and co are 1hz
#ARM QC、：“will these data be trusted enough to use”
#    obj = arm.read_netcdf(files)
#    diff = obj[var].diff('time', n=1)
#
#    time = obj['time'].values
#
#    ds = diff.to_dataset(name='cpc_diff')
#    ds['cpc_diff'].values = abs(diff.values)
#    ds = ds.rolling(time=period, min_periods=minp,center=True).mean()'
# %% Training part
cpc = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.20171111*')
co = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.2017111*')

cpc_diff = cpc['concentration'].diff('time', n=1)
cpc_time = cpc['time'].values
cpc_ds = cpc_diff.to_dataset(name='cpc_diff')

cpc_ds['cpc_diff'].values = abs(cpc_diff.values)
cpc_ds = cpc_ds.rolling(time=period, min_periods=minp,center=True).mean()

#%%
sdate = '20171111'
edate = sdate
#files = glob.glob(file_path)
#ds,cpc = getAOSData(files)
#time = ds['time'].values

stime=['010738','011133']
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
    #%%
#Create the flag that we want to train for.. I.e. exhaust, no exhaust
y_train=np.zeros(len(cpc_time))
all_indices=[]
for i in range(len(sbad)):
    idx=(cpc_time >= sbad[i])*(cpc_time <= ebad[i])
    all_indices.append(list(np.where(idx)[0]))

#Set indices of previous periods to bad
y_train[all_indices[0]]=1.
#%%

#Setup the model using 16 cores
#Random_state=0 gaurantees that we will get the same result each time
model = RandomForestClassifier(n_estimators = nest,max_depth=md,random_state=0,min_samples_leaf=leafs,n_jobs=16)

#Fit the model  to the training dataset
model.fit(cpc_ds.to_dataframe(),y_train[:-1])
#%%
try:
    fi=model.feature_importances_
    col=ds.columns
    print(col)
    print(fi)
    type='RandomForest'
except:
    type='KNeighbors'
#%
#%% Test part

files = glob.glob('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201*')
files.sort()
files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]

cpc_test = arm.read_netcdf(files)
cpc_diff_test = cpc_test['concentration'].diff('time', n=1)
cpc_time_test = cpc_test['time'].values

cpc_test_ds = cpc_diff_test.to_dataset(name='cpc_diff')
cpc_test_ds['cpc_diff'].values = abs(cpc_diff_test.values)
cpc_test_ds = cpc_test_ds.rolling(time=period, min_periods=minp,center=True).mean()
#%%
# Show summary
cpc_con = cpc['concentration'].to_dataframe(name = 'cpc')
cpc_con.describe()
co_con = co['co_dry'].to_dataframe(name = 'co')
co_con.describe()
#%%
#Get Data to apply the ML algorith to
args = sys.argv
if len(args) > 1:
    sdate = str(args[1])
    edate = sdate
if len(args) > 2:
    edate = str(args[2])
sdate = '20171029'
edate = '20180324'

#%% Training data
#files = glob.glob(files_path)
#files.sort()
#files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]
#
#obs_ds,cpc = getAOSData(files)
#
#X_test = obs_ds.to_dataframe()
# %%
'''parallel'''
#    Working method uses the probabilites that the model ouputs that a point is exhaust
#    Smooths that data out so we don't get noisy flagging every other time
prob=model.predict_proba(cpc_con)[: ,1]
prob=pd.DataFrame(data=prob).rolling(min_periods=5, window=60*10 ,center=True).mean().values.flatten()
#
#    Flags anything with a probability higher than 1.5% which seems very small
#    but actually works out very well
fflag=0.00001
#    fflag=0.9999999
idx=(prob >= fflag)
idx = np.append(idx,True)
index=np.where(idx)[0]

#    E = int(1.5*N)
#    S = int(0.5*N)
#The rest of the program is plotting up the data for visualization and testing purposes
time=cpc_con.index.to_pydatetime
#%% Figure show result of ml
plt.plot(cpc_con)
plt.plot(cpc_con.iloc[index[:-1]])

