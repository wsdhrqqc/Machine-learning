#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:16:31 2020
1113,1114 to train
@author: qingn
"""

import matplotlib as mpl
#mpl.use('Agg')
from matplotlib.dates import DateFormatter,date2num
from collections import Counter
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
from sklearn import preprocessing


co_path = '/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.2017111[3-4]*'
o3_path = '/Users/qingn/Desktop/NQ/maraoso3/maraoso3M1.b1.2017111[3-4]*.custom.nc'
cpc_path = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2017111[3-4]*'

co = arm.read_netcdf(co_path)

o3 = arm.read_netcdf(o3_path)

cpc = arm.read_netcdf(cpc_path)
#%%
cpc_con = cpc['concentration']#.resample(time='1min').mean()
cpc_con = cpc_con.resample(time='1s').nearest()
#   obj = arm.read_netcdf(files,variables=var)
co = arm.read_netcdf(co_path)
o3 = arm.read_netcdf(o3_path)
co_con = co['co_dry'].where(co['qc_co_dry']<16384)
co_con = co_con.resample(time='1s').nearest()
co_diff = co_con.diff('time', n=1)

o3_con = o3['o3'].where(o3['qc_o3']<262144)
o3_con = o3_con.resample(time='1s').nearest()
# o3_con = o3_con.resample(time = '1min').mean()
o3_diff = o3_con.diff('time', n=1)


ds_ori = xr.Dataset({"co":(("time"),co_con),
                 "o3":(("time"),o3_con),
                 # "cpc":(("time"),cpc_con)
                 },
                coords={"time":o3_con.time},)
ds_ori = ds_ori.rolling(time=15, min_periods=5,center=True).mean()
ds_ori = ds_ori.dropna(dim = 'time')


co_ori_nor = preprocessing.normalize([ds_ori.co.data])
o3_ori_nor = preprocessing.normalize([ds_ori.o3.data])

ds_ori['co_nor'] = co_ori_nor[0]
ds_ori['o3_nor'] = o3_ori_nor[0]



#%% Set Train
sdate = '20171113'
edate = sdate

time = ds_ori.time[:-64864]

stime=['032630','050950']#,'014342']
etime=['033227','052225']#,'022022']
sbad=[]
ebad=[]

for i in range(len(stime)):
    sdummy=np.datetime64(''.join(['-'.join([sdate[0:4],sdate[4:6],sdate[6:8]]),
        'T',':'.join([stime[i][0:2],stime[i][2:4],stime[i][4:6]])]))
    sbad.append(sdummy)
    edummy=np.datetime64(''.join(['-'.join([edate[0:4],edate[4:6],edate[6:8]]),
        'T',':'.join([etime[i][0:2],etime[i][2:4],etime[i][4:6]])]))
    ebad.append(edummy)
    
sdate = '20171114'
edate = sdate
stime=['014342']
etime=['022022']
sbad_=[]
ebad_=[]


for i in range(len(stime)):
    sdummy=np.datetime64(''.join(['-'.join([sdate[0:4],sdate[4:6],sdate[6:8]]),
        'T',':'.join([stime[i][0:2],stime[i][2:4],stime[i][4:6]])]))
    sbad_.append(sdummy)
    edummy=np.datetime64(''.join(['-'.join([edate[0:4],edate[4:6],edate[6:8]]),
        'T',':'.join([etime[i][0:2],etime[i][2:4],etime[i][4:6]])]))
    ebad_.append(edummy)

sbad = sbad+sbad_
ebad = ebad+ebad_
#%%
#Create the flag that we want to train for.. I.e. exhaust, no exhaust
y_train=np.zeros(len(time))
all_indices=[]
for i in range(len(sbad)):
    idx=(time >= sbad[i])*(time <= ebad[i])
    all_indices.append(list(np.where(idx)[0]))

#Set indices of previous periods to bad
y_train[all_indices[0]]=1.
y_train[all_indices[1]]=1.
y_train[all_indices[2]]=1.

#Sets different values for the RandomForestClassifier
md=5 #Max depth of branches
nest=25  #Total number of trees in the forest
leafs=5 # Min number of leafs to use
#%%
#Setup the model using 
#Random_state=0 gaurantees that we will get the same result each time
model = RandomForestClassifier(n_estimators = nest,
                               max_depth=md,random_state=0,
                               min_samples_leaf=leafs,n_jobs=16)

#Fit the model  to the training dataset
d = {'co':ds_ori.co_nor[:-64864],'o3':ds_ori.o3_nor[:-64864],'time':time}
df = pd.DataFrame(data=d)
df = df.set_index('time')
model.fit(df,y_train)

#%% Test
#Get Data to apply the ML algorith to
sdate = '20171029'
edate = '20180324'

# co_full = 
# o3_full = 
co_path = '/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.201*'
o3_path = '/Users/qingn/Desktop/NQ/maraoso3/maraoso3M1.b1.201*.custom.nc'
co_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.20*'))
o3_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraoso3/maraoso3M1.b1.20*.custom.nc'))
#%% 'read test'
# co = arm.read_netcdf(co_path)
co = arm.read_netcdf(co_files[32:44])
_, index1 = np.unique(co['time'], return_index = True)
co = co.isel(time = index1)

# o3 = arm.read_netcdf(o3_path)
o3 = arm.read_netcdf(o3_files[23:35])
_, index1 = np.unique(o3['time'], return_index = True)
o3 = o3.isel(time = index1)

co_con = co['co_dry'].where((co['qc_co_dry']<16384) & (co['qc_co_dry']!=34)& (co['qc_co_dry']!=44)& (co['qc_co_dry']!=28))
co_con = co_con.resample(time='1s').nearest()
# co_diff = co_con.diff('time', n=1)
o3_con = o3['o3'].where(o3['qc_o3']<260000)
# o3_con = o3['o3'].where((o3['qc_o3']==0)|(o3['qc_o3']==8)|(o3['qc_o3']==2056)|(o3['qc_o3']==3072))
o3_con = o3_con.resample(time='1s').nearest()
# o3_con = o3_con.resample(time = '1min').mean()
# o3_diff = o3_con.diff('time', n=1)
#%% co is not clean
# o3_ = o3.where(o3['qc_o3']<262144)
# o3_con = o3['o3'].where(o3['qc_o3']<262144)
# o3_.qc_o3[np.where(o3_con<0)[0]].values

# plt.plot(o3.o3[np.where(o3.qc_o3==2)[0]])
# co_ = co.where(co['qc_co_dry']<16384)
# co_.qc_co_dry[np.where(co_con<-0.1)[0]].values
#%% set up test
ds_ori = xr.Dataset({"co":(("time"),co_con),
                 "o3":(("time"),o3_con)
                 # "cpc":(("time"),cpc_con)
                 },
                coords={"time":o3_con.time},)
ds_ori = ds_ori.rolling(time=15, min_periods=5,center=True).mean()
ds_ori = ds_ori.dropna(dim = 'time')
time = ds_ori.time

co_ori_nor = preprocessing.normalize([ds_ori.co.data])
o3_ori_nor = preprocessing.normalize([ds_ori.o3.data])

ds_ori['co_nor'] = co_ori_nor[0]
ds_ori['o3_nor'] = o3_ori_nor[0]

d_ = {'co':ds_ori.co_nor.values,'o3':ds_ori.o3_nor.values,'time':time.values}
df_ = pd.DataFrame(data=d_)
X_test = df_.set_index('time')
#%
#    Working method uses the probabilites that the model ouputs that a point is exhaust
#    Smooths that data out so we don't get noisy flagging every other time
prob=model.predict_proba(X_test)[: ,1]
# OR

prob=pd.DataFrame(data=prob).rolling(min_periods=5 ,window=60*10 ,center=True).mean().values.flatten()
#%%
#    Flags anything with a probability higher than 1.5% which seems very small
#    but actually works out very well
fflag=0.4
#    fflag=0.9999999
idx=(prob >= fflag)
idx = np.append(idx,True)
index=np.where(idx)
#%%
pred=model.predict(X_test)
pred_f=pd.DataFrame(data=pred).rolling(min_periods=20 ,window=120*10 ,center=True).mean().values.flatten()
df_['flag'] = pred


#    E = int(1.5*N)
#    S = int(0.5*N)
#The rest of the program is plotting up the data for visualization and testing purposes
time=X_test.index.to_pydatetime()
#%%
#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
#%%
cpc_full_p = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2*'))
cpc_full = arm.read_netcdf(cpc_full_p[24:36])

# _, index1 = np.unique(cpc_full['time'], return_index = True)
# cpc_full = cpc_full.isel(time = index1)
#%%
index_cpc_ml = np.array(index[0])
flg = np.zeros(len(time))
flg[index_cpc_ml-1] = 1
df_['flag'] =flg
df_ = df_.set_index('time')
# df_['flag'] = 0
# df_['flag'][index]
# cpc_full['flag'] = flg
# df_.index=pd.to_datetime(df_.index)
df_10 = df_.resample('10min').mean()
#%%
_, index1 = np.unique(cpc_full['time'], return_index = True)
cpc_full = cpc_full.isel(time = index1)
# qc_con = cpc_full['qc_concentration']
cpc_con = cpc_full['concentration']
# cpc_con = cpc_con.where((qc_con!=2)&(qc_con!=65474)&(qc_con!=962))
cpc_con_10 = cpc_con.resample(time = '1min').mean().resample(time = '10min').nearest()
df_10['cpc'] = cpc_con_10[1:]
#%%
plt.plot(df_10.index,df_10.o3,'r.')
plt.plot(df_10.index[df_10['flag']>0.1],df_10.o3[df_10['flag']>.1],'b.')

#%%
plt.plot(df_10.index,df_10.co,'r.')
plt.plot(df_10.index[df_10['flag']>0.1],df_10[df_10['flag']>.1],'b.')
#%%
plt.plot(cpc_con_10[1:].time,cpc_con_10[1:],'k.')

plt.plot(df_10.index[df_10['flag']>0.0],df_10['cpc'][df_10['flag']>0.0],'r.')
#%%
idx1 = find_closest(time.values, cpc_con_10.time.values)
flag_ml = cpc_full.flag[idx1]
index_ml = np.array(np.where(flag_ml==1)[0])