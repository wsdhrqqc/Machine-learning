#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:56:53 2020
WIBS4 DATA (Waveband Integrated Bioaerosol Sensor mark4)
FBAP: Fluorescent Biological Aerosol Particl
@author: qingn
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime

FIGWIDTH = 12
FIGHEIGHT = 4 
FONTSIZE = 22
LABELSIZE = 22
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}
from scipy import optimize
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.cm as cm
from scipy.stats import linregress
plt.rcParams['font.family'] = ['Times New Roman']
#%%
wibs = pd.read_csv('/Users/qingn/Desktop/NQ/WIBS_MARCUS_NC_3std_noport.csv', parse_dates=True)
exhaust_wibs = pd.read_csv('/Users/qingn/Desktop/NQ/WIBS_MARCUS_NC_3std_noport_exhausfilter.csv', parse_dates=True)
#%%
time=pd.to_datetime(wibs['Date_Time'])
#date_time= datetime.strptime(wibs['Date_Time'][0], '%d.%m.%Y %H:%M:%S')
date_time1= pd.to_datetime(wibs['Date_Time'], format = '%d.%m.%Y %H:%M:%S')
ex_time1=pd.to_datetime(exhaust_wibs['Date_Time'], format = '%d.%m.%Y %H:%M:%S')
#%%
#date_format = mdates.DateFormatter('%D')
fig, ax0 = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(12, 8))    
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
ax = ax0.twinx()
#days = mdates.DayLocator()
#days_fmt = mdates.DateFormatter('%D')


#ax0.xaxis.set_major_formatter(date_format)
##ax0.xaxis.set_major_locator(ticker.MultipleLocator(5))
#ax0.xaxis.set_major_locator(days)
#ax0.xaxis.set_major_formatter(days_fmt)
#ax0.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#ax0.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#ax0.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax0.tick_params(axis='both',which='major',labelsize=12,direction='in')
#ax0.tick_params(axis='both',which='minor',direction='in')


ax0.format_xdata = mdates.DateFormatter('%m-%d-%H')
#ax0.plot(date_time1[:1500],wibs['FBAP'][:1500],label = 'ori')
#ax.plot(time[:1500],wibs['FBAP'][:1500],'.r',label = 'ori')
ax.plot(ex_time1[:],exhaust_wibs['FBAP'][:],'b^',label = 'ex')
ax.plot(ex_time1[:],exhaust_wibs['NonF'][:],'g-',label = 'NonF')


ax0.plot(ex_time1[:],exhaust_wibs['FBAP'][:]*100/exhaust_wibs['All'][:1500],'r*',label = 'ratio')
#ax0.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax0.grid(True)
ax.set_ylabel('N_FBAP(#/L)')
ax0.set_ylabel('N_fbap/Total(%)')
ax0.legend(loc=(0.01, 0.75), fontsize=16)
ax.legend(loc=(0.01, 0.15), fontsize=16)
ax.set_yscale("log")
ax0.set_yscale("log")


fig.autofmt_xdate()
#%% Scatter plot of 'All' and wind speed
clean_boole = np.array(df_try_10min['new_flag']==0)
plt.plot(df_try_10min['All'][clean_boole])
plt.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['All'][clean_boole])


small_wind = np.logical_and(df_try_10min['wind_speed']>0.0, df_try_10min['wind_speed']<10)
middle_wind = np.logical_and(df_try_10min['wind_speed']>10.0, df_try_10min['wind_speed']<20)
huge_wind = np.logical_and(df_try_10min['wind_speed']>20.0, df_try_10min['wind_speed']<35)

clean_small =  np.logical_and(clean_boole, small_wind )
clean_middle =  np.logical_and(clean_boole, middle_wind )
clean_huge =  np.logical_and(clean_boole, huge_wind )

#%%
plt.scatter(df_try_10min['wind_speed'][clean_small],df_try_10min['All'][clean_small],color = 'r',label = 'weak');
plt.scatter(df_try_10min['wind_speed'][clean_middle],df_try_10min['All'][clean_middle],color = 'b',label = 'middle');
plt.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['All'][clean_huge],color = 'g',label = 'huge')

#%%
plt.scatter(df_try_10min['wind_speed'][clean_small],df_try_10min['cpc_con'][clean_small],color = 'r',label = 'weak');
plt.scatter(df_try_10min['wind_speed'][clean_middle],df_try_10min['cpc_con'][clean_middle],color = 'b',label = 'middle');
plt.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['cpc_con'][clean_huge],color = 'g',label = 'huge')
#%%
plt.scatter(df_try_10min['wind_speed'][clean_small],df_try_10min['accumulation'][clean_small],color = 'r',label = 'weak');
plt.scatter(df_try_10min['wind_speed'][clean_middle],df_try_10min['accumulation'][clean_middle],color = 'b',label = 'middle');
plt.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['accumulation'][clean_huge],color = 'g',label = 'huge')

#%%
plt.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['accumulation'][clean_boole],color = 'g',label = '.1-1.0um')
plt.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['All'][clean_boole],color = 'b',label = '.5<..<10um')
plt.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['cpc_con'][clean_boole],color = 'r',label = '>0.01um')
#plt.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['FBAP'][clean_boole],color = 'y',label = 'FBAP')
#plt.yscale('log')
plt.legend()
#%%
#plt.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['accumulation'][clean_huge],color = 'g',label = '.1-1.0um')
plt.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['All'][clean_boole],color = 'y',label = 'FBAP .5<..<10um')
#plt.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['cpc_con'][clean_huge],color = 'b',label = '.8<..<16um')
#plt.scatter(df_try_10min['wind_speed'][huge_wind],df_try_10min['FBAP'][huge_wind],color = 'y',label = 'FBAP')
plt.yscale('log')
plt.xlabel('speed m/s')
plt.ylabel('#/L')
plt.legend(loc = 'upper left')
#%%
fig1, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(9, 5))
#fig1, ax1 = 
ax1 = ax.twinx()

ax.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['accumulation'][clean_boole]-0.1,color = 'g',label = '.1-1.0um')
#ax.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['All'][clean_boole],color = 'b',label = '.5<..<10um')
#ax1.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['FBAP'][clean_boole],color = 'y',label = 'All_martin')
ax1.scatter(df_ori_try_10min_['wind_speed'][df_ori_try_10min_['new_flag_x']==0],df_ori_try_10min_['All'][df_ori_try_10min_['new_flag_x']==0],label = '.5<..<10um')
ax.scatter(df_try_10min['wind_speed'][clean_boole],df_try_10min['cpc_con'][clean_boole],color = 'r',label = '>.01um')

Data = df_try_10min[clean_boole][['wind_speed','accumulation']].dropna()
A2,B2  = optimize.curve_fit(f_1,Data['wind_speed'],Data['accumulation'])[0]
y4= A2 * Data['wind_speed']+ B2

ax.plot(Data['wind_speed'],y4,color = 'g',linewidth = 1.5,linestyle = '-',zorder =2)

Data = df_try_10min[clean_boole][['wind_speed','cpc_con']].dropna()
A2,B2  = optimize.curve_fit(f_1,Data['wind_speed'],Data['cpc_con'])[0]
y4= A2 * Data['wind_speed']+ B2
ax.plot(Data['wind_speed'],y4,color = 'r',linewidth = 1.5,linestyle = '-',zorder =2)


ax.set_ylabel('Con #/cc')
ax1.set_ylabel('"All" #/L')
ax1.yaxis.label.set_color('b')
#ax1.set_ylabel('FBAP #/L')
ax1.legend(loc = 'upper left')
ax.legend(loc = 'upper right')
ax.set_xlabel('wind speed(m/s)')
ax.set_yscale('log')
#ax.set_ylim((.1,10000))
#ax1.set_yscale('log')

#plt.ylabel('c')
#%%
import seaborn as sns
sns.set(color_codes=False)
sns.regplot(x="wind_speed", y="accumulation", data=df_try_10min[clean_boole],logx=True);

#%%
fig1, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(12, 4))
#fig1, ax1 = 
ax1 = ax.twinx()

#ax.scatter(df_try_10min['wind_speed'][clean_middle],df_try_10min['accumulation'][clean_middle],color = 'g',label = '.1-1.0um')
#ax.scatter(df_try_10min['wind_speed'][clean_middle],df_try_10min['cpc_con'][clean_middle],color = 'r',label = '>.01um')
#ax.scatter(df_try_10min['wind_speed'][clean_middle],df_try_10min['All'][clean_middle],color = 'b',label = '.5<..<10um')
#ax1.scatter(df_try_10min['wind_speed'][middle_wind],df_try_10min['FBAP'][middle_wind]-170,color = 'y',label = 'FBAP')

ax.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['accumulation'][clean_huge],color = 'g',label = '.1-1.0um')
ax.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['cpc_con'][clean_huge],color = 'r',label = '>.01um')
ax.scatter(df_try_10min['wind_speed'][clean_huge],df_try_10min['All'][clean_huge],color = 'b',label = '.5<..<10um')
ax1.scatter(df_try_10min['wind_speed'][huge_wind],df_try_10min['FBAP'][huge_wind],color = 'y',label = 'FBAP')

ax.set_ylabel('con #/L or #/cc')
ax1.set_ylabel('FBAP #/L')
#ax1.legend(loc = 'upper left')
ax.legend(loc = 'upper right')
ax.set_xlabel('wind speed(m/s)')
#%% correlation between accu and N_CCN

#df_10min copied the df_try_10min and changed the large value
N = sum(np.logical_and(df_10min['new_flag']==True , df_10min['SS']==0.2))
#df_accu_ccn = pd.DataFrame()
#df_accu_ccn['N_CCN'] = 
df_accu_ccn = df_10min[np.logical_and(df_10min['new_flag']==True , df_10min['SS']==0.2)][['N_CCN','accumulation']].dropna()
x = df_accu_ccn['N_CCN']
y = df_accu_ccn['accumulation']
#x = df_10min['N_CCN'][np.logical_and(df_10min['new_flag']==True , df_10min['SS']==0.2)]
#y = df_10min['accumulation'][np.logical_and(df_10min['new_flag']==True , df_10min['SS']==0.2)]

C = round(r2_score(x,y),4)
rmse = round(np.sqrt(mean_squared_error(x,y)),3)
x2 = np.linspace(0,400)
y2=x2
def f_1(x,A,B):
    return A*x+B
A1,B1  = optimize.curve_fit(f_1,x,y)[0]
y3= A1*x +B1

fig , ax = plt.subplots(1,2,figsize=(8,3),dpi=200,sharey=True, facecolor = 'white')

ax[0].scatter(x,y,edgecolor=None,c = 'k',s =5, marker='s')
ax[0].plot(x2,y2,color = 'k',linewidth = 1.5,linestyle = '-',zorder =2)
ax[0].plot(x,y3,color = 'r',linewidth = 1.5,linestyle = '-',zorder =2)

# 添加上限和下限
#绘制upper line
up_y2 = 1.15*x2 + 0.05
#绘制bottom line
down_y2 = 0.85*x2 - 0.05
ax[0].plot(x2,up_y2,color='g',lw=1,ls='--',zorder=2)
ax[0].plot(x2,down_y2,color='g',lw=1,ls='--',zorder=2)

fontdict1 = {"size":12,
             "color":'k',}
ax[0].set_xlabel("N_CCN(#/cc)",fontdict = fontdict1)
ax[1].set_xlabel("N_CCN(#/cc)",fontdict = fontdict1)
ax[0].set_ylabel("accumulation mode(#/cc)",fontdict = fontdict1)
ax[0].grid(False)
ax[0].set_xticks(np.arange(0,500,step = 150))
ax[0].set_yticks(np.arange(0,500,step = 150))
fontdict1 = {"size":10,
             "color":'k',}
ax[0].text(10,400,r'$R^2=$'+str(round(C,3)),fontdict = fontdict1)
ax[0].text(10,350,'RMSE='+str(rmse),fontdict = fontdict1)
ax[0].text(10,300,r'$y=$'+str(round(A1,3))+'$x$'+" + "+str(round(B1,3)),fontdict = fontdict1)
ax[0].text(10,250,r'$N=$'+str(N),fontdict = fontdict1)

nbins = 150
H,xedges,yedges = np.histogram2d(x,y,bins = nbins)
H = np.rot90(H)
H = np.flipud(H)
Hmasked = np.ma.masked_where(H==0,H)

plt.pcolormesh(xedges, yedges, Hmasked, cmap=cm.get_cmap('gist_rainbow'),vmin =0,vmax=20)
ax[1].set_xticks(np.arange(0,500,step = 150))
ax[1].set_yticks(np.arange(0,500,step = 150))
ax[1].tick_params(left = True,bottom = True, direction='in')

cbar = plt.colorbar(ax=ax[1],ticks = [0,5,10,15,20],drawedges = False)
colorbarfontdict = {"size":9,"color":'k'}
cbar.ax.set_title('Counts',fontdict = colorbarfontdict,pad = 8)
cbar.ax.tick_params(labelsize=10,direction='in')
cbar.ax.set_yticklabels(['0','5','10','15','>20'],family = 'Times New Roman')

slope = linregress(x,y)[0]
intercept = linregress(x,y)[1]
lmfit = (slope*x)+intercept
ax[1].plot(x,lmfit,c='r',linewidth=1.5)
ax[1].plot([0,450],[0,450],c='k',ls='-',zorder=1,lw=1.5)
ax[1].plot(x2,up_y2,c='b',linewidth=1,ls = '--',zorder=2)
ax[1].plot(x2,down_y2,c='b',linewidth=1,ls = '--',zorder=2)

ax[1].text(10,400,r'$R^2=$'+str(round(C,3)),fontdict = fontdict1)
ax[1].text(10,350,'RMSE='+str(rmse),fontdict = fontdict1)
ax[1].text(10,300,r'$y=$'+str(round(A1,3))+'$x$'+" + "+str(round(B1,3)),fontdict = fontdict1)
ax[1].text(10,250,r'$N=$'+str(N),fontdict = fontdict1)
#ax[1].set_xlabel("N_CCN(#/cc)",fontdict = fontdict1)
fig.suptitle('ss=0.2%')
#%% distribution
#np.logical_and(df_aero_['wind_speed']>0.0, df_try_10min['wind_speed']<10)

dff_uhsas['wind_flag'][np.logical_and(dff_uhsas['wind_speed']>0 , dff_uhsas['wind_speed']<6)] = 0
dff_uhsas['wind_flag'][np.logical_and(dff_uhsas['wind_speed']>6 , dff_uhsas['wind_speed']<12)] = 1
dff_uhsas['wind_flag'][np.logical_and(dff_uhsas['wind_speed']>12 , dff_uhsas['wind_speed']<18)] = 2
dff_uhsas['wind_flag'][np.logical_and(dff_uhsas['wind_speed']>18 , dff_uhsas['wind_speed']<24)] = 3
plt.hist(dff_uhsas['wind_flag'])
#%%

#distribution2 = dff_uhsas[np.logical_and(dff_uhsas['wind_flag']==2,dff_uhsas[]].sum()/2493
distribution2 = dff_uhsas[np.logical_and(dff_uhsas['wind_flag']==2,clean_boole)].sum()/646#2493
distribution0 = dff_uhsas[np.logical_and(dff_uhsas['wind_flag']==0,clean_boole)].sum()/559#1677
distribution1 = dff_uhsas[np.logical_and(dff_uhsas['wind_flag']==1,clean_boole)].sum()/1421#4144
distribution3 = dff_uhsas[np.logical_and(dff_uhsas['wind_flag']==3,clean_boole)].sum()/210#496
#%%
plt.plot(lower_bd[:-1]*0.001,distribution0[-99:-1],label = '0-6m/s',color = 'purple')
plt.plot(lower_bd[:-1]*0.001,distribution1[-99:-1],label = '6-12m/s',color = 'orange')
plt.plot(lower_bd[:-1]*0.001,distribution2[-99:-1],label = '12-18m/s',color = 'blue')
plt.plot(lower_bd[:-1]*0.001,distribution3[-99:-1],label = '18-24m/s',color = 'green')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('dn/dlnDp(1/cc/um)')
plt.xlabel('Dp(um)')
plt.legend()
