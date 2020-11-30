#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:50:36 2019

@author: qingn
"""

import matplotlib as mpl
#mpl.use('Agg')
from matplotlib.dates import DateFormatter,date2num

import act.io.armfiles as arm
import act.plotting.plot as armplot
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
# from sklearn.neighbors import NearestNeighbors
# %%

#if __name__ == '__main__':
   
datastream = 'maraoscpc'
var = 'concentration'


def getAOSData(files):
    period = 15
    minp = 5

#   obj = arm.read_netcdf(files,variables=var)
    obj = arm.read_netcdf(files)
    diff = obj[var].diff('time', n=1)

    time = obj['time'].values

    ds = diff.to_dataset(name='cpc_diff')
    ds['cpc_diff'].values = abs(diff.values)
    ds = ds.rolling(time=period, min_periods=minp,center=True).mean()
#    ds = ds['cpc_diff'].resample(time='1S').mean()
    #ds['cpc'] = obj[var]

    return ds, obj[var]

# %%
#diff_cpc = np.ma.ediff1d(cpc_con_ms)
#diff_cpc = np.append(diff_cpc,0)
#diff_cpc = abs(diff_cpc)
# %%
def machine_learning(file_path,files_path):
    #Use ADC example script to get the data
    # ''.join(['./',datastream,'/*',sdate,'*'])
    sdate = '20171111'
    edate = sdate
    files = glob.glob(file_path)
    ds,cpc = getAOSData(files)
    time = ds['time'].values

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