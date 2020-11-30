clc;clear;
nav_path = '/Users/qingn/Desktop/NQ/marnav/marnavbeM1.c1.20171029.000000.nc';
cpc_path = '/Users/qingn/Desktop/NQ/maraoscpc/maraoscpcfM1.b1.20171030.000000.nc';
co_path  = '/Users/qingn/Desktop/NQ/maraosco/maraoscoM1.b1.20171030.190000.nc';
exhaust_path = '/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc';
cpc = ncread(cpc_path,'concentration');
nav = ncread(nav_path);
nav_info = ncinfo(nav_path);
cpc_time = ncread(cpc_path,'time');
cpc_info = ncinfo(cpc_path);
co = ncread(co_path,'co');
co_time = ncread(co_path,'time');
exhaust_id = ncread(exhaust_path,'exhaust_4mad02thresh');
exhaust_id_time = ncread(exhaust_path,'time');

% 
