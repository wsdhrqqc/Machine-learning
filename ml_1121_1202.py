#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:07:09 2020

@author: qingn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:11:35 2020

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

ds_ori.o3.loc['2017-11-13 11:34:9':'2017-11-13 11:34:18'] = 23.828
# ds_ori = pd.concat([ds_ori[:'2017-11-13 11:33:55'],ds_ori['2017-11-13 11:34:33':]])

ds_ori = ds_ori.rolling(time=30, min_periods=5,center=True).mean()
ds_ori = ds_ori.dropna(dim = 'time')


# co_ori_nor = preprocessing.normalize([ds_ori.co.data])
# o3_ori_nor = preprocessing.normalize([ds_ori.o3.data])

# ds_ori['co_nor'] = co_ori_nor[0]
# ds_ori['o3_nor'] = o3_ori_nor[0]
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
#Sets different values for the RandomForestClassifier
md=5 #Max depth of branches
nest=25  #Total number of trees in the forest
leafs=5 # Min number of leafs to use

#Setup the model using 16 cores
#Random_state=0 gaurantees that we will get the same result each time
model = RandomForestClassifier(n_estimators = nest,
                               max_depth=md,random_state=0,
                               min_samples_leaf=leafs,n_jobs=16)

#%%

#%%
#Create the flag that we want to train for.. I.e. exhaust, no exhaust
y_train=np.zeros(len(ds_ori.co[:-64864]))
all_indices=[]
for i in range(len(sbad)):
    idx=(time >= sbad[i])*(time <= ebad[i])
    all_indices.append(list(np.where(idx)[0]))

#Set indices of previous periods to bad
y_train[all_indices[0]]=1.
y_train[all_indices[1]]=1.
y_train[all_indices[2]]=1.


#Fit the model  to the training dataset
d = {'co':ds_ori.co[:-64864],'o3':ds_ori.o3[:-64864],'time':ds_ori.co.time[:-64864]}
df = pd.DataFrame(data=d)
df = df.set_index('time')


X_scaled = preprocessing.scale(df)

scaler = preprocessing.StandardScaler().fit(X_scaled)
scaler.scale_

# scaler.transform(X_scaled)

# X_train = scaler.transform()

model.fit(X_scaled,y_train)
#%% Test
#Get Data to apply the ML algorith to
sdate = '20171029'
edate = '20180324'

# co_full = 
# o3_full = 
# co_path = '/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.201*'
# o3_path = '/Users/qingn/Desktop/NQ/maraoso3/maraoso3M1.b1.201*.custom.nc'
co_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.20*'))
o3_files = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraoso3/maraoso3M1.b1.20*.custom.nc'))

#%% 'read test- slow'
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
#%% set up test
ds_ori = xr.Dataset({"co":(("time"),co_con),
                 "o3":(("time"),o3_con)
                 # "cpc":(("time"),cpc_con)
                 },
                coords={"time":o3_con.time},)
ds_ori = ds_ori.rolling(time=30, min_periods=5,center=True).mean()
ds_ori = ds_ori.dropna(dim = 'time')


# co_ori_nor = preprocessing.normalize([ds_ori.co.data])
# o3_ori_nor = preprocessing.normalize([ds_ori.o3.data])

# ds_ori['co_nor'] = co_ori_nor[0]
# ds_ori['o3_nor'] = o3_ori_nor[0]

d_ = {'co':ds_ori.co.values,'o3':ds_ori.o3.values,'time':ds_ori.time.values}
df_ = pd.DataFrame(data=d_)
X_test = df_.set_index('time')

X_test_scaled = preprocessing.scale(X_test)
# X_test_scaled = scaler.transform(X_test)
time=X_test.index.to_pydatetime()

pred=model.predict(X_test_scaled)
pred_f=pd.DataFrame(data=pred).rolling(min_periods=5 ,window=240*10 ,center=True).mean().values.flatten()

df_['flag'] = pred_f
df_ = df_.set_index('time')
#%%
cpc_full_p = sorted(glob.glob('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2*'))
cpc_full = arm.read_netcdf(cpc_full_p[24:36])

cpc_train =arm.read_netcdf('/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.2017111[3-4]*')
cpc_train = cpc_train['concentration'][:-64864]
cpc_train_10 = cpc_train.resample(time = '1min').mean().resample(time = '10min').nearest()
#%% slow
df_10 = df_.resample('10min').mean()
_, index1 = np.unique(cpc_full['time'], return_index = True)
cpc_full = cpc_full.isel(time = index1)
# qc_con = cpc_full['qc_concentration']
cpc_con = cpc_full['concentration']
# cpc_con = cpc_con.where((qc_con!=2)&(qc_con!=65474)&(qc_con!=962))
cpc_con_10 = cpc_con.resample(time = '1min').mean().resample(time = '10min').nearest()
df_10['cpc'] = cpc_con_10[1:]
#%% Varification
plt.plot(df_10.index,df_10.co,'r.')
plt.plot(df_10.index[df_10['flag']>0.1],df_10[df_10['flag']>.1].co,'b.')
#%%
plt.plot(df_10.index,df_10.o3,'r.')
plt.plot(df_10.index[df_10['flag']>0.0],df_10[df_10['flag']>.0].o3,'b.')
#%%
fig = plt.figure()
ax1 = plt.subplot(311)
ax1.plot(cpc_con_10[1:].time,cpc_con_10[1:],'k.',markersize=3)

ax1.plot(df_10.index[df_10['flag']>0.0],df_10['cpc'][df_10['flag']>0.0].values,'r.',markersize=3)
ax1.set_yscale('log')
ax1.set_ylabel('con(#/cc)')

ax2= plt.subplot(313,sharex=ax1)
ax2.plot(df_10.index,df_10.co,'r.',markersize=3)
ax2.plot(df_10.index[df_10['flag']>0.1],df_10[df_10['flag']>.1].co,'b.',markersize=2)
ax2.set_ylabel('co(ppmv)')
ax2.set_yscale('log')

ax3 = plt.subplot(312,sharex=ax1)
ax3.plot(df_10.index,df_10.o3,'r.',markersize=3)
ax3.plot(df_10.index[df_10['flag']>0.1],df_10[df_10['flag']>.1].co,'b.',markersize=3)
ax3.set_ylabel('o3(ppbv)')
fig.autofmt_xdate()

fig.savefig('ml_cpc_co_o3.png',dpi=450)
#%%
plt.plot(pred_f)
#%% Train
plt.plot(df.index,df.co,'k.')
plt.plot(df.index[y_train==1],df.co[y_train==1],'r.')

#%% Train
plt.plot(df.index,df.o3,'k.')
plt.plot(df.index[y_train==1],df.o3[y_train==1],'r.')
#%% Train
# df_train = df_10['2017-11-13':'2017-11-14 05:47:15']
df_train = df_10['2017-11-13':'2017-11-13 12:47:15']
# plt.plot(cpc_train_10.time,cpc_train_10.values,'k.')
# plt.plot(df_train[df_train['flag']>0].index,df_train[df_train['flag']>0].cpc)
plt.plot(df_train['2017-11-13':'2017-11-13 16:47:15'].index,df_train['2017-11-13':'2017-11-13 16:47:15'].cpc)
#%%
df_10.to_csv('ml_co_o3_flag_1121-1203.csv')