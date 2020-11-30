#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:02:32 2019
Creating Qing's flag for MARCUS aerosol after using functions
@author: qingn
"""
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib.dates import DateFormatter,date2num

import act.io.armfiles as arm
import act.plotting.plot as armplot
#import act.discovery.get_files as get_data
from sklearn.pipeline import Pipeline
from pipeline_components import DataFrameSelector, DataScaler, DataSampleDropper

#from sklearn.preprocessing import StandardScaler

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

datastream = 'maraoscpc'
var = 'concentration'
period = 10
minp = 5

#%%

def getdata(address):
    obj = arm.read_netcdf(address)
    time = obj['time']
    qc_cpc = obj['qc_concentration']
    cpc_con = obj[var]
    diff = obj[var].diff('time', n=1)
    diff = np.append(diff,diff[-1])
    df = pd.DataFrame({'cpc_con':cpc_con,'time':time,'diff_con':abs(diff)})
    df = df.set_index('time')
    #df['diff_con'] = df['cpc_con'].diff('time', n=1)
    df['cpc_con'] = df['cpc_con'].rolling(period, min_periods=minp,center=True).mean()
    df['diff_con'] = df['diff_con'].rolling(period, min_periods=minp,center=True).mean()
    missing = np.where((qc_cpc==2)|(qc_cpc==962)|(qc_cpc==65474))[0]
    df['cpc_con'].values[missing]= np.NaN
    df['diff_con'].values[missing] = np.NaN
    
    return df
df = getdata('./maraoscpc/maraoscpcfM1.b1.20171111*.nc')
    #%%

'''

address = './maraoscpc/maraoscpcfM1.b1.20171111*.nc'
obj = arm.read_netcdf(address)

time = obj['time']
qc_cpc = obj['qc_concentration']
cpc_con = obj[var]
diff = obj[var].diff('time', n=1)
diff = np.append(diff,diff[-1])
df = pd.DataFrame({'cpc_con':cpc_con,'time':time,'diff_con':abs(diff)})
df = df.set_index('time')
#df['diff_con'] = df['cpc_con'].diff('time', n=1)
df['cpc_con'] = df['cpc_con'].rolling(period, min_periods=minp,center=True).mean()
df['diff_con'] = df['diff_con'].rolling(period, min_periods=minp,center=True).mean()

#pd.DataFrame({'time':time[:-1],'cpc_diff':diff, 'cpc_con':obj[var][:-1]}, index = time[:-1])
#diff = obj[var].diff('time',n = 1)
#df = diff.to_dataframe(name:cpc_diff)
#df = diff.to_dataframe(name='diff_cpc')
#df['cpc_con']=obj[var][:-1]
#df['diff_cpc']=abs(diff)
# delete everything qc!=0 except the value larger than alarming and max valid
missing = np.where((qc_cpc==2)|(qc_cpc==962)|(qc_cpc==65474))[0]
df['cpc_con'].values[missing]= np.NaN
df['diff_con'].values[missing] = np.NaN
'''
#%%
'''PRE-PROCESS DATA'''

selected_features = df.columns
scaled_features = ['cpc_con','diff_con']


pipe = Pipeline([
    ('RowDropper', DataSampleDropper()),
    ('FeatureSelector', DataFrameSelector(selected_features)),
    ('Scale', DataScaler(scaled_features))
])

processed_data = pipe.fit_transform(df)# TODO
print(processed_data.isnull().values.any())
# PLot the training data
fig= plt.figure(figsize = (15,5))
myFmt = DateFormatter("%H:%M:%S")
ax = fig.gca()

ax.xaxis.set_major_formatter(myFmt); 

ax.plot(processed_data['cpc_con'][4000:4400],'.',linewidth = 1.0, color = 'grey', label = 'ori_data')
ax.plot(processed_data['diff_con'][4000:4400],'.',linewidth = 1.0, color = 'black', label = 'ori_diff')
# 

#%%
'''We are actually creating our training data with Nov 11 data'''
sdate = '20171111'
edate = sdate
#files = glob.glob('./maraoscpc/maraos*.nc')
#obj = arm.read_netcdf(files)
# Load data
#cpc = arm.read_netcdf('./maraoscpc/maraoscpcfM1.b1.20171111*.nc')
#processed_data_train = new_getAOSData('./maraoscpc/maraoscpcfM1.b1.20171111*.nc')
#trytry = df['cpc_con'].rolling(period, min_periods=minp,center=True).mean()
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

#missing = np.where((qc_cpc==2)|(qc_cpc==962)|(qc_cpc==65474)[0])
#ones = np.ones(np.size(time))
#ones[missing]=0
#left = np.where(ones==1)
time = processed_data.index
y_train=np.zeros(np.size(time))
all_indices=[]

for i in range(len(sbad)):
    idx=(time >= sbad[i])&(time <= ebad[i])
    all_indices.append(list(np.where(idx)[0]))
    
#Set indices of previous periods to bad
    y_train[all_indices[i]]=1.

   
#%%
'''Set up of the RF model'''
#Sets different values for the RandomForestClassifier
md=500 #Max depth of branches
nest=25  #Total number of trees in the forest
leafs=10 # Min number of leafs to use

#Setup the model using 16 cores
#Random_state=0 gaurantees that we will get the same result each time
model = RandomForestClassifier(n_estimators = nest,max_depth=md,random_state=0,min_samples_leaf=leafs,n_jobs=16)

#Fit the model  to the training dataset
X_train = processed_data.astype('float64')
model.fit(X_train,y_train)
print(model.feature_importances_)
#Get Data to apply the ML algorith to
args = sys.argv
if len(args) > 1:
    sdate = str(args[1])
    edate = sdate
if len(args) > 2:
    edate = str(args[2])
sdate = '20171029'
edate = '20180324'
#df['time'] = time[:-1]
#df.set_index('time')
#ds,cpc = getAOSData(files)
prob=model.predict_proba(X_train)[: ,1]
prob=pd.DataFrame(data=prob).rolling(min_periods=5 ,window=60*10 ,center=True).mean().values.flatten()
plt.hist(prob)
fflag=0.0001
#    fflag=0.9999999
idx=(prob >= fflag)
#idx = np.append(idx,True)

index=np.where(idx)[0]
print(np.size(index)/np.size(prob))
time_train = processed_data.index

cpc_coo_train = X_train['cpc_con']
#maraoscpc/maraoscpcfM1.b1.201711*.nc
#cpc = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.20171123*.nc')
#cpc_con = cpc['concentration']
#time_cpc = cpc['time']
fig= plt.figure(figsize = (10,5))
myFmt = DateFormatter("%m/%d-%H")
ax = fig.gca()
N = 86400
i = 0

#ax.plot(cpc_test,label = 'original')
#ax.plot(cpc_test[index], label = 'bad')
ax.xaxis.set_major_formatter(myFmt); 

ax.plot(cpc_coo_train,'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
ax.plot(cpc_coo_train[index],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')
plt.plot(cpc_coo_train[np.where(y_train==1)[0]],color = 'orange')
#plt.plot(time_train[np.where(y_train==1)[0]],cpc_coo_train[np.where(y_train==1)[0]],color = 'orange')
#%%
files = glob.glob('./maraoscpc/maraoscpcfM1.b1.201*.nc')
files.sort()
files = [f for f in files if f.split('.')[-3] >= sdate and f.split('.')[-3] <= edate]

df_test= getdata(files)
#%%
df_test= getdata('./maraoscpc/maraoscpcfM1.b1.201*.nc')

#X_test1 = obs_ds.to_dataframe()

#df_test=getdata('./maraoscpc/maraoscpcfM1.b1.2017111*.nc')
#%%
'''
address = './maraoscpc/maraoscpcfM1.b1.2017111*.nc'
obj_test = arm.read_netcdf(address)

time = obj_test['time'].values
qc_cpc = obj_test['qc_concentration']
cpc_con = obj_test[var]
diff = obj_test[var].diff('time', n=1)
diff = np.append(diff,diff[-1])
df_test = pd.DataFrame({'cpc_con':cpc_con,'time':time,'diff_con':abs(diff)})


#df_test = pd.DataFrame({'cpc_con':cpc_con,'time':time})
df_test.set_index('time')
#pd.DataFrame({'time':time[:-1],'cpc_diff':diff, 'cpc_con':obj[var][:-1]}, index = time[:-1])
#diff = obj[var].diff('time',n = 1)
#df = diff.to_dataframe(name:cpc_diff)
#df = diff.to_dataframe(name='diff_cpc')
#df['cpc_con']=obj[var][:-1]
#df['diff_cpc']=abs(diff)

missing = np.where((qc_cpc==2)|(qc_cpc==962)|(qc_cpc==65474))[0]
df_test['cpc_con'].values[missing]= np.NaN
df_test['diff_con'].values[missing] = np.NaN
'''
#%%
'''PRE-PROCESS DATA'''

selected_features = df_test.columns
scaled_features = ['cpc_con','diff_con']


pipe = Pipeline([
    ('RowDropper', DataSampleDropper()),
    ('FeatureSelector', DataFrameSelector(selected_features)),
    ('Scale', DataScaler(scaled_features))
])
#    pipe2 = Pipeline([('RowDropper', DataSampleDropper())])
processed_data_test = pipe.fit_transform(df_test)# TODO
print(processed_data_test.isnull().values.any())

X_test = processed_data_test.astype('float64')
#X_test = processed_data_test.drop(['time'], axis=1).astype('float64')

prob=model.predict_proba(X_test)[: ,1]
prob=pd.DataFrame(data=prob).rolling(min_periods=5 ,window=60*10 ,center=True).mean().values.flatten()
plt.hist(prob)
fflag=0.0001
#    fflag=0.9999999
idx=(prob >= fflag)
#idx = np.append(idx,True)
index=np.where(idx)[0]
print(np.size(index)/np.size(prob))
#%%
time_test = processed_data_test.index

cpc_coooon = X_test['cpc_con']
#maraoscpc/maraoscpcfM1.b1.201711*.nc
#cpc = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.20171123*.nc')
#cpc_con = cpc['concentration']
#time_cpc = cpc['time']
fig= plt.figure(figsize = (12,5))
myFmt = DateFormatter("%m/%d-%H")
ax = fig.gca()
N = 86400
i = 0

#ax.plot(cpc_test,label = 'original')
#ax.plot(cpc_test[index], label = 'bad')
ax.xaxis.set_major_formatter(myFmt); 
#for i in [0,1,2,3,4]:
#    E = int((i+1.5)*1.0*N)
#    S = int((i)*1.0*N)
#    fig= plt.figure(figsize = (10,5))
#    ax = fig.gca()
#    ax.plot_date(time_test[S:E], cpc_coooon[S:E],'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
#    ax.plot_date(time_test[index[(index>S)&(index<E)]], cpc_coooon[index[(index>S)&(index<E)]],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')

#
ax.plot_date(time_test, cpc_coooon,'.',linewidth = 1.0, color = 'grey', label = 'be_left_data')
#    
ax.plot_date(time_test[index], cpc_coooon[index],'.',linewidth = 0.6, color = 'blue', alpha = 0.5,label = 'mad_contaminated')

#%%
#Save file
indate = 2017103020180324
cpc_wind_fullname = '/Users/qingn/%sqing_flag_00.cdf'%indate
df_test['exhaust_flag'] = 0
df_test['exhaust_flag'][index]= 1
df_test = df_test.to_xarray()
df_test.attrs['location_description'] = 'Measurements of Aerosols, Radiation and CloUds over the Southern Ocean (MARCUS), Resupply Vessel Aurora Australis'
df_test.attrs['input_data'] = 'ARM data streams: maraoscpcfM1.a1*'
df_test.attrs['summary'] = 'OU Qing'' Flag 1.0 version(qingniu@edu.au for more information).'
df_test.attrs['sampling_interval'] = '1 second'
df_test.attrs['title'] = 'Exhaust identification product for MARCUS project aboard Aurora Australis 2017/18'
df_test.attrs['timezone']='UTC'
#win_cpc_df1 = pd.DataFrame({'exhaust_flag':idx}, index = df_test.index)
#joblib.dump(win_cpc_df1, cpc_wind_fullname)
#result = df_test['exhaust_flag']
#df_test.to_csv(cpc_wind_fullname)

df_test.to_netcdf(cpc_wind_fullname)
