###
# VTGが0.5より大きい時間を対象に，
# ４h以上連続している箇所を４つピックアップして，
# GGA, VTG(COG, SOG), VBW, VHWを描写
###

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
from tqdm import tqdm
import pickle as pkl
import math
import os 
import os.path as osp
import re

import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)

from utils import *
from kf_params import *

path = path_ship = r"E:\shunsukeE\data\shiplog/"

year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month)
done_ship = []

target_ships  = {'黒潮': '黒潮丸'}
def latlon_to_mesh_df(lat, lon, deg_per_mesh=1/36, size=[1050, 1191], latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    grid0 = ((lon-latlon_range[1]).astype(int)/deg_per_mesh)
    grid1 = ((lat-latlon_range[0]).astype(int)/deg_per_mesh)
    grid0[grid0 < 0] = -1
    grid0[grid0 > size[0]] = -1
    grid1[grid1 < 0] = -1
    grid1[grid1 > size[1]] = -1
    return grid0, grid1

def mean_ground_speed(dt, lat1, lon1, lat2, lon2):
    
    lat1 = dms_to_deg(lat1)
    lon1 = dms_to_deg(lon1)
    
    lat2 = dms_to_deg(lat2)
    lon2 = dms_to_deg(lon2)
    
    dist, deg = dist_deg_latlon(lat1, lon1, lat2, lon2)
    speed = dist/dt
    return ((lat1+lat2)/2, (lon1+lon2)/2), (speed, deg)

def dist_deg_latlon(lat1, lon1, lat2, lon2):
    from geopy.distance import geodesic
    res = geodesic((lat1, lon1), (lat2, lon2))
    
    # 方位角を計算
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    delta_lon = lon2_rad - lon1_rad
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(y, x))

    # 0から360度の範囲に調整
    bearing = (bearing + 360) % 360
    return res.meters, bearing

def dms_to_deg(x):
    deg = x // 100
    mit = x - deg*100
    #print(x)
    #print(deg)
    #print(mit)
    #print(sec)
    return deg + mit/60

def deg_to_rat(deg):
    #  北基準のdef
    return (-deg+90)*np.pi/180

def remove_outlier(df, key):
    # 下位・上位５％のデータを消す（外れ値対策）
    q1 = df[key].quantile(0.05)
    q2 = df[key].quantile(0.95)
    tf = (df[key]>q1) & (df[key]<q2)
    df2 = df[tf]
    return df2
    
def latlon_knot(lat1, lon1, lat2, lon2):
    distance = haversine_distance(lat1, lon1, lat2, lon2)/60
    knot = (distance/(time_set[i+1]-time_set[i])) * 1.94384
    return knot

def divide_nmea_voyage(log_data):
    # 時間の整理　dtIdx: 0時からの経過時間，dtIdx_Minute:0時0分からの経過分
    time_utc =log_data["UTC"].values
    str_format = '%Y/%m/%d %H:%M:%S'
    epoc_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
    time_idx = []
    time_idx2 = []
    tf = []
    for t in time_utc:
        idx = datetime.datetime.strptime(t, str_format) 
        idx2 = idx - epoc_dt
        time_idx.append(int(idx2.days*24 + idx2.seconds / (60*60)))
        time_idx2.append(int(idx2.days*24*60 + idx2.seconds / (60)))
    time_idx = np.array(time_idx)
    time_idx2 = np.array(time_idx2)
    log_data["DtIdx"] = time_idx
    log_data["DtIdx_Minute"] = time_idx2

    # NMEAデータの抽出
    # ggaデータの抽出
    gga = log_data[log_data["NMEA"]=="GGA"]
    gga.columns = ['UTC', 'NMEA', 'LatDMS', 'LonDMS', 'DtIdx', 'DtIdx_Minute']
    gga['Lat'] = dms_to_deg(gga['LatDMS'])
    gga['Lon'] = dms_to_deg(gga['LonDMS'])

    # vbwデータの抽出
    vbw = log_data[log_data["NMEA"]=="VBW"]
    vbw.columns = ['UTC', 'NMEA', 'LonWaterSpeed', 'TraWaterSpeed', 'DtIdx', 'DtIdx_Minute']

    
    # hdtデータの抽出
    hdt = log_data[log_data["NMEA"]=="HDT"]
    hdt = hdt.drop('data2', axis=1)
    hdt.columns = ['UTC', 'NMEA', 'HeadDeg', 'DtIdx', 'DtIdx_Minute']

    # vtgデータの抽出    
    vtg = log_data[log_data["NMEA"]=="VTG"]
    vtg.columns = ['UTC', 'NMEA', 'HeadDeg', 'GroundSpeed', 'DtIdx', 'DtIdx_Minute']

    # vhwデータの抽出
    vhw = log_data[log_data["NMEA"]=="VHW"]
    vhw.columns = ['UTC', 'NMEA', 'HeadDeg', 'WaterSpeed', 'DtIdx', 'DtIdx_Minute']
    
    tf = vtg['GroundSpeed']>0.5
    vtg2 = vtg[tf]
    time = sorted(set(vtg2['DtIdx'].values))

    def split_into_ranges(lst):
        ranges = []
        current_range = [lst[0]]

        for i in range(1, len(lst)):
            if lst[i] - lst[i-1] == 1:
                current_range.append(lst[i])
            else:
                ranges.append(current_range)
                current_range = [lst[i]]

        ranges.append(current_range)
        return ranges

    if len(time)==0:
        splited_time = []
    else:
        splited_time = split_into_ranges(time)
    
    return gga, vbw, hdt, vtg, vhw, splited_time

for target_ship in target_ships.keys():
    print(f'{target_ship} ({target_ships[target_ship]})')
    if target_ship in done_ship:
        print(f'skip {target_ship}')
        continue

    shipLog = pd.DataFrame([])
    for day in range(4, 4+1):
        f_path = osp.join(path_ship, target_ship, '2015', f'{target_ships[target_ship]}FileOut{year}{month:02}{day:02}.slog1')
        try:
            log = pd.read_csv(f_path, encoding="cp932", header=None)
            log.columns = ["UTC", "NMEA", "data1", "data2"]
            shipLog = pd.concat([shipLog, log])
        except:
            print(f'Error load {f_path}')
    if len(shipLog)==0:
        print(f'Not exist data')
        continue
    
    gga, vbw, hdt, vtg, vhw,  splited_time = divide_nmea_voyage(shipLog)

    if len(splited_time)==0:
        continue
    count = 0
    max_voyage_num = 4
    sns.set(style="whitegrid", palette="muted", color_codes=True)
    plt.rcParams['font.family'] = 'MS Gothic' 
    fig, axes = plt.subplots(nrows=max_voyage_num, ncols=5, figsize=(4*5, 4*max_voyage_num))
    for i in range(len(splited_time)):
        if len(splited_time[i])<4:
            continue
        ax = axes[count, 0]
        ax.set_title(f'GGA voyage-{i}')
        gga1 = gga[gga['DtIdx'].isin(splited_time[i])]
        if len(gga1)>0:
            gga1.plot.scatter(x='Lon', y='Lat',
                marker='s', c='b', s=1, alpha=0.5, ax=ax)

        ax = axes[count, 1]
        ax.set_title(f'VTG (COG) voyage-{count}')
        vtg1 = vtg[vtg['DtIdx'].isin(splited_time[i])]
        if len(vtg1)>0:
            vtg1.plot.scatter(x='DtIdx_Minute', y='HeadDeg',
                marker='s', c='b', s=1, alpha=0.5, ax=ax)          
       
        ax = axes[count, 2]
        ax.set_title(f'VTG (SOG) voyage-{count}')
        if len(vtg1)>0:
            vtg1.plot.scatter(x='DtIdx_Minute', y='GroundSpeed',
                marker='s', c='b', s=1, alpha=0.5, ax=ax) 

        ax = axes[count, 3]
        ax.set_title(f'HDT (HDT) voyage-{count}')
        hdt1 = hdt[hdt['DtIdx'].isin(splited_time[i])]
        if len(hdt1)>0:
            hdt1.plot.scatter(x='DtIdx_Minute', y='HeadDeg',
                marker='s', c='b', s=1, alpha=0.5, ax=ax)  

        ax = axes[count, 4]
        ax.set_title(f'VBW, VHW (STW) voyage-{count}')
        vbw1 = vbw[vbw['DtIdx'].isin(splited_time[i])]
        if len(vbw1)>0:
            vbw1.plot.scatter(x='DtIdx_Minute', y='LonWaterSpeed',
                marker='s', c='b', s=1, alpha=0.5, ax=ax, label='VBW')

        vhw1 = vhw[vhw['DtIdx'].isin(splited_time[i])]
        if len(vhw1)>0:
            vhw1.plot.scatter(x='DtIdx_Minute', y='WaterSpeed',
                marker='s', c='g', s=1, alpha=0.5, ax=ax, label='VHW')  
        count += 1
        if count>=max_voyage_num:
            break
    plt.title(f'voyage-{target_ship}')
    plt.subplots_adjust(wspace=0.3, hspace=0.3) 
    fname = f'voyage-test.png'
    save_path = osp.join(path_ship, target_ship, '2015', fname)
    plt.savefig(save_path)
    plt.close('all')
    print(f'saved {save_path}')

