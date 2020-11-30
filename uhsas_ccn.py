#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 05:12:05 2019

@author: qingn
"""

import xarray as xr
import dask
import numpy as np
import matplotlib.backends.backend_pdf
import scipy.stats as stats
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
matplotlib.use('Agg')
import sys
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
#import mpl_toolkits
#import mpl_toolkits.basemap as bm
#from mpl_toolkits.basemap import Basemap, cm
import act
import seaborn as sns
#import module_ml
#from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier

import pathlib
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 18
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


#matplotlib.rc('xtick', labelsize=LABELSIZE) 
#matplotlib.rc('ytick', labelsize=LABELSIZE) 
HOME_DIR = str(pathlib.Path.home()/'Desktop/NQ' )
#%% I use this function to pick up subset from a very large range in exhaust_id
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def arm_read_netcdf(directory_filebase, time_resolution):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time = time_resolution).nearest()
    return file

def arm_read_netcdf2(directory_filebase):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>'''
    
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
#    file = file_ori.resample(time = time_resolution).nearest()
    file = file_ori
    return file

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
#%%
# EXHAUST_ID
exhaust_id = netCDF4.Dataset('/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc')
exhaust = exhaust_id['exhaust_4mad02thresh']
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))   

# UHSAS
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201711*.nc'
uhsas = arm_read_netcdf(path_uhsas,'10min')

#_, index1 = np.unique(uhsas['time'], return_index = True)
#uhsas = uhsas.isel(time = index1)
#uh_con = uhsas['concentration'][:,19:]
con_new = uhsas['concentration'][:,19:].sum(axis=1) # accu+coarse
con_new1 = uhsas['concentration'][:,19:62].sum(axis=1) # accu

'''size_distribution / (sample_flow_rate/60.0 * sample_accumulation_time)'''
#uhsas_con = uhsas['size_distribution']/(uhsas['sample_flow_rate']*uhsas['sample_accumulation_time'])
#uh_con = uhsas_con[:,75:]
#uh_con = uhsas_con[:,19:]
time_uhsas = uhsas['time'].values

#%% CPC
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.201711*'
cpc = arm_read_netcdf(path_cpc ,'1h')
cpc_con = cpc['concentration']
time_cpc = cpc['time'].values
#%%

#/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1.20171104.001708.nc
#ccn_colavg_1 = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosccn/maraosccn1colavgM1.b1.2017111*')
path_ccn = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.201711*.nc'

ccn_colavg = arm_read_netcdf2(path_ccn)
#_, index1 = np.unique(ccn_colavg['time'], return_index = True)
#ccn_colavg = ccn_colavg.isel(time = index1)
con = ccn_colavg['N_CCN']
qc_con = ccn_colavg['qc_N_CCN']
#con = con[np.where(qc_con==0)[0]]
time_ccn = ccn_colavg['time'].values

#%%

idx2 = find_closest(time_id_date, time_ccn) 
#idx1 = find_closest(time_id_date, time_uhsas)


flag_ccn = exhaust[idx2] # to create a relative smaller exhaust_id array
'''This step is slow!!'''
'''
editted
'''
#%%
a = []
for i in np.arange(np.size(idx2)):
    a.append(sum(exhaust[idx2[i]:(idx2[i]+3600)]))
#    print(sum(exhaust[idx2[i]:(idx2[i]+3600)]))
aarray = np.asarray(a)
clean_in_hour = np.where(aarray<360)
idx3 = idx2[clean_in_hour]
flag_ccn_hour = exhaust[idx3]
'''This step is slow!!'''

index_ccn_mad = np.where(flag_ccn_hour == 1)# pick up the contaminated time index
dirty2 = np.array(index_ccn_mad[0]) # name the index to be dirty(contaminated index)
index_ccn_clean_mad = np.where(flag_ccn_hour == 0)
clean2 = np.array(index_ccn_clean_mad[0])# name the index to be clean(clean index)
# %% 
# implement flag on uhsas total concentration(1)
#idx1 = find_closest(time_id_date, time_uhsas)
idx1 = find_closest(time_uhsas, time_ccn[clean2])
#flag_uhsas = exhaust[idx1]
uhsas_at_ccn_time_accu_corse = con_new[idx1]
uhsas_at_ccn_time_accu = con_new1[idx1]
'''This step is slow!!'''

ccn_con01 = np.ma.masked_where(qc_con[clean2,1]!= 0, con[clean2][:,1]) 
ccn_con02 = np.ma.masked_where(qc_con[clean2,2]!= 0, con[clean2][:,2])
ccn_con05 = np.ma.masked_where(qc_con[clean2,3]!= 0, con[clean2][:,3])
ccn_con08 = np.ma.masked_where(qc_con[clean2,4]!= 0, con[clean2][:,4])
ccn_con10 = np.ma.masked_where(qc_con[clean2,5]!= 0, con[clean2][:,5])    
dataframe_ccn = pd.DataFrame({'accumulation_coarse_mode':uhsas_at_ccn_time_accu_corse,'accumulation_mode':uhsas_at_ccn_time_accu,'con_at_0.1%':ccn_con01,'con_at_0.2%':ccn_con02,'con_at_0.5%':ccn_con05,'con_at_0.8%':ccn_con08,'con_at_1.0%':ccn_con10},index = time_ccn[clean2])
dataframe_ccn_qc = pd.DataFrame({'accumulation_coarse_mode':uhsas_at_ccn_time_accu_corse,'accumulation_mode':uhsas_at_ccn_time_accu,'con_at_0.1%_qc':ccn_con01,'con_at_0.1%':con[clean2][:,1],'con_at_0.2%_qc':ccn_con02,'con_at_0.2%':con[clean2][:,2],'con_at_0.5%_qc':ccn_con05,'con_at_0.5%':con[clean2][:,3],'con_at_0.8%_qc':ccn_con08,'con_at_0.8%':con[clean2][:,4],'con_at_1.0%_qc':ccn_con10,'con_at_1%':con[clean2][:,5]},index = time_ccn[clean2])
# %%
print(stats.pearsonr(uhsas_at_ccn_time_accu,con[clean2][:,3]))
sns.set(color_codes=True)


#sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6,"xtick.major.size": 6, "xtick.major.size": 6})

rc={'axes.labelsize': 18, 'font.size': 18, 'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize': 18, 'axes.titlesize': 18}
sns.set(rc)
g = sns.jointplot(x = 'accumulation_coarse_mode',y = 'con_at_0.5%',data = dataframe_ccn, kind = 'reg', stat_func=r2)
g.set_axis_labels('accumulation_mode(#/cc)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 20)# %%

        #%%print(stats.pearsonr(uhsas_at_ccn_time_accu,con[clean2][:,5]))
sns.set(color_codes=Truesns.set(color_codes=True))#,xlim=[30,1000],ylim=[0.01,300])


#sns.set_style("ticks", {"xtick.major.size": 6, "ytick.major.size": 6,"xtick.major.size": 6, "xtick.major.size": 6})

rc={'axes.labelsize': 18, 'font.size': 18, 'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize': 18, 'axes.titlesize': 18}
sns.set(rc)
g = sns.jointplot(x = 'accumulation_mode_new',y = 'con_at_0.5%',data = a, kind = 'reg', stat_func=r2)
g.set_axis_labels('accumulation_mode(#/cc)','CCN concentration(#/cc)',fontweight = 'bold',fontsize = 20)
#
#%%
index_uhsas_mad = np.where(flag_uhsas == 1)# pick up the contaminated time index
dirty1 = np.array(index_uhsas_mad[0])
index_uhsas_clean_mad = np.where(flag_uhsas == 0)
clean1 = np.array(index_uhsas_clean_mad[0])
#%%
clean_ratio = np.size(clean2)/np.size(flag_ccn)
print('cpc clean ratio is',clean_ratio)
qc_zero_ratio = np.size(np.where(qc_con[clean2,4]==0))/np.size(qc_con[clean2,4])
print('ccn qc==0 ratio is',qc_zero_ratio)
#%%
lower_bd = uhsas['lower_size_limit'].values[0,]
upper_bd = uhsas['upper_size_limit'].values[0,]
#interval = upper_bd-lower_bd
interval_meta=uhsas['upper_size_limit'][0,] - uhsas['lower_size_limit'][0,]
con_non_nan = np.nan_to_num(uh_con) # meta automatically dispeared
total_con = np.dot(con_non_nan, interval_meta[19:])

#total_con = np.dot(con_non_nan, interval_meta[74:]) #unit:1/cc
#
#lower_bd = np.append(lower_bd,[1000.])
#dlnDp = np.log(lower_bd[1:]-lower_bd[:-1])
#dn_dlnDp_all = uh_con/dlnDp[3:]

#%% ## Figure time series
myFmt = DateFormatter("%m/%d-%H")
nrows = 2
ncols = 1
N = 480*3
M = 24*3
ss_ratio = [0.1,0.2,0.5,0.8,1.0]
fig = plt.figure(figsize = (FIGWIDTH*2.2,FIGHEIGHT*1.2))
#ax = plt.axes([0., 0., 1., .8]
#fig = plt.figure()
for x in [1,2,3,4,5]:
    ss = ss_ratio[x-1]
    for i in np.arange(3): # total 144days 48 figures
    #for i in [27,28,29,30]:
        S1 = int((i)*1.0*N) # for uhsas
        E1 = S1+N
    #    S = S_+14400
        S = int(i*M)
        E = S+M
        
    #    fig, axs = plt.subplots(nrows,ncols,figsize = (FIGWIDTH*3.3,FIGHEIGHT*2),sharex =True)
        
        p = 0
        ax = fig.gca()
    #    fig.subplots_adjust(wspace=.15, hspace=.3)
        ax.plot_date(time_uhsas[dirty1[(dirty1>S1)&(dirty1<E1)]],total_con[dirty1[(dirty1>S1)&(dirty1<E1)]],
                                          '.',linewidth = 0.6, color = 'yellow', alpha = 0.5,label = 'mad_uhsas contaminated')
        ax.plot_date(time_uhsas[clean1[(clean1>S1)&(clean1<E1)]],total_con[clean1[(clean1>S1)&(clean1<E1)]],
                                          '.',linewidth = 0.6, color = 'green', alpha = 0.5,label = 'mad_uhsas clean')
        
        
        ax.plot(time_ccn[dirty2[(dirty2>S)&(dirty2<E)]],con[:,x][dirty2[(dirty2>S)&(dirty2<E)]],
                                          '*',linewidth = 0.6, color = 'red', alpha = 0.5,label = 'mad_ccn contaminated')
        ax.plot(time_ccn[clean2[(clean2>S)&(clean2<E)]],con[:,x][clean2[(clean2>S)&(clean2<E)]],
                                          '*',linewidth = 1, color = 'blue',label = 'mad_ccn clean')
      
        
        ax.legend(markerscale=3,ncol=4, bbox_to_anchor=(0.5, -0.2),
                  loc=10, fontsize='small')
    #    ax.legend(markerscale=3)
        ax.set_title('N_CCN(supersaturation_setpoint_'+str(ss)+'%)')
        ax.xaxis.set_major_formatter(myFmt); 
        ax.yaxis.grid()
        ax.set_ylabel('1/cc')
        #axs[p].set_ylim((0,25))
        ax.set_yscale('log')
        fdir=HOME_DIR +'/'+'uhsas_ccn_comp_figure'
        try:
            os.stat(fdir)
        except:
            os.mkdir(fdir)
        print('Writing: '+fdir+'uhsas_ccn'+str(i)+'_'+str(i+3)+'.png')
        plt.tight_layout()
    #    plt.show()
    #    plt.gcf()
#        plt.savefig(fdir+'/uhsas_ccn_'+str(ss)+'%_'+str(i)+'_'+str(i+3)+'.png')  
        plt.cla()
    #    pdf.savefig(fig)
    
    
plt.close('all')
#%% 

con[:,5].values[np.where(qc_con[:,5]>0)[0]] = np.nan
uh_ccn_df = pd.DataFrame({'uhsas':total_con[clean2], 'CCN':con[:,5][clean2]}, index = time_ccn[clean2])

plt.title('boxplot for uhsas&ccn')
uh_ccn_df.boxplot()
plt.ylabel('counts per cc')
#%% Figure scatter plot


#N = 480*3
M = 24*3
ss_ratio = [0.0,0.1,0.2,0.5,0.8,1.0]
for x in [0,1,2,3,4,5]:
    ss = ss_ratio[x]
    win_cpc_df1 = pd.DataFrame({'uhsas':total_con[clean2], 'CCN':con[:,x][clean2]}, index = time_ccn[clean2])
#    fig,ax = plt.subplots(figsize=(6,6))
 # can also get the figure from plt.gcf()
    g = sns.JointGrid('uhsas', 'CCN', win_cpc_df1)
    g.plot_marginals(sns.distplot, hist=True, kde=True, color='blue')
    g.plot_joint(plt.scatter, color='black', edgecolor='black')
    ax = g.ax_joint
    plt.plot(total_con[clean2],total_con[clean2], c = 'blue')
#    plt.scatter(total_con[clean2], con[:,x][clean2])
#    plt.plot(total_con[clean2],total_con[clean2], c = '.3')
#    plt.xlabel('UHSAS accumulation mode concentration(1/cc)')
    ax.plot([0,0],[6000,6000],ls = '-')
    g.set_axis_labels('UHSAS accumulation mode concentration(1/cc)','N_CCN(1/cc)', fontweight = 'bold',fontsize = 18)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('CCN V.S. UHSAS when ss='+str(ss)+'%')
    plt.xlim((0,500))
    plt.ylim((0,500))
#    plt.ylabel('N_CCN(1/cc)')
    
#    plt.title('UHSAS VS CCN when ss='+str(ss))
#    im = ax.hexbin(total_con[clean2], con[:,x][clean2], gridsize=20, 
#               cmap=plt.cm.BuGn)
#%%
#win_cpc_df1 = pd.DataFrame({'uhsas':total_con[clean2], 'CCN':con[:,4][clean2]}, index = time_ccn[clean2])

#%% Density SCatter
sns.set(font_scale = 1.5)

g = sns.jointplot(x = 'uhsas', y = 'CCN', data = uh_ccn_df, xlim=(0,1000), ylim=(0,1000),kind ="kde", color="lightcoral")
g.set_axis_labels('UHSAS(#/cc)','CCN concentration(#/cc)')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('whole MARCUS voyage')
#plt.title('compare concentration(#/cc)')
#ax.set_xscale('log')
#ax.set_yscale('log')
#%% Point Sctatter
g = sns.JointGrid('uhsas', 'CCN', win_cpc_df1)
g.plot_marginals(sns.distplot, hist=True, kde=True, color='blue')
g.plot_joint(plt.scatter, color='black', edgecolor='black')
ax = g.ax_joint
plt.plot(total_con[clean2],total_con[clean2], c = 'blue')
#ax.set_xscale('log')
#ax.set_yscale('log')
#g.ax_marg_x.set_xscale('log')
#g.ax_marg_y.set_yscale('log')
#%% Figure spectra
path_ccn1 = '/Users/qingn/Desktop/NQ/marccnspectra/maraosccn1colspectraM1.b1.2017111[7-9]*.nc'
ccn_colavg1 = arm_read_netcdf(path_ccn1,'1h')
time_ccn1 = ccn_colavg1['time']
con_ccn1 = ccn_colavg1['N_CCN']

fig = plt.figure(figsize = (FIGWIDTH*2.2,FIGHEIGHT*1.2))
ax = fig.gca()
for x in [0,1,2,3,4]:
    ax.plot(time_ccn1,con_ccn1[:,x],label = 'ss = %s'%(ss_ratio[x])+'%')
#    ax.legend(markerscale=3,ncol=4, bbox_to_anchor=(0.5, -0.2),
#          loc=10, fontsize='small')
    ax.legend(markerscale=3)
    ax.set_title('Number of CCN concentration(spectra)')
    ax.xaxis.set_major_formatter(myFmt); 
    ax.xaxis.grid()
    ax.set_ylabel('#/cc_log')
    #axs[p].set_ylim((0,25))
    ax.set_yscale('log')