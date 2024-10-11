
###
# x軸を各船の偏流，y軸をAISもしくはJCOPEの偏流として図を作成する
###
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
import tqdm
import pickle as pkl
import math
import os 
import os.path as osp
import re

from utils import *
from kf_params import *



year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month)
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_ship = path = r"E:\shunsukeE\data\shiplog/"
save_path = r'./Images/test'
target_ships  = {'中春': '中春丸',
                '2辰巳': '第二辰巳丸',
                '11和光': '第十一和光丸',
                '18英山': '第十八英山丸',
                '21東': '第二十一東丸',
                '33東洋': '第三十三東洋丸',
                '87東洋': '第八十七東洋丸',
                'MARS': 'SUNNYMARS',
                'ひま2': 'ひまわり２',
                '興春': '興春丸',
                '黒潮': '黒潮丸',
                '昇山': '昇山丸',
                '昭建': '昭建丸',
                '昭瑞': '昭瑞丸',
                '清栄': '清栄丸',
                '双信': '双信丸',
                '筑前': '筑前丸',
                '如月': '如月丸',
                '八菱': '第八菱洋丸',
                '豊鶴': '豊鶴丸',
                '立眞': '立眞丸'}

def kuroshio_ndata(shipLog):
    ndata_map = np.zeros(nan_map.shape)

    for i in range(len(shipLog)):
        grid0 = shipLog['Grid0'].values[i]
        grid1 = shipLog['Grid1'].values[i]
        ndata_map[grid0][grid1] += 1
    ndata = kurosio_filter(ndata_map, nan_map)
    x = np.arange(len(ndata))
    plt.bar(x, ndata)
    plt.title('Num data of ship log in kuroshio')
    plt.xlabel('Index')
    plt.ylabel('Num data')
    fpath = osp.join(save_path, f"ndata_shiplog2.png")
    plt.savefig(fpath)
    pkl.dump(ndata_map, open('data/shiplog_ndata.pkl', 'wb'))

def save_map_shipLog(shipLog, save_path=None):
    m = np.zeros((24, map_size_ais[0], map_size_ais[1]))
    for i in range(len(shipLog)):
        time = shipLog['DtIdx'].values[i]
        grid0 = shipLog['Grid0'].values[i]
        grid1 = shipLog['Grid1'].values[i]
        curN = shipLog['CurN'].values[i]
        curE = shipLog['CurE'].values[i]
        dtidx = time
        #if dtidx<dt_range[0] or dt_range[1]<dtidx:
        if dtidx<0 or 23<dtidx:
            print(f'dtidx: {dtidx} is out of range. (range: {dt_range})')
            continue
        if grid0<0 or grid0>map_size_ais[0]:
            print(f'grid0: {grid0} is out of range. (range: {grid0})')
            continue
        if grid1<0 or grid1>map_size_ais[1]:
            print(f'grid1: {grid1} is out of range. (range: {grid1})')
            continue
        m[time][grid0][grid1] = 1.0

    for i in range(24):
        dt = datetime.datetime(dt_year, dt_month, dt_day, i, 0, 0)
        #plt.plot(m[i])
        sns.heatmap(m[i]*kurosio_map)
        fpath = osp.join(save_path, f"map_shipLog{dt.year}{dt.day:02}{dt.hour:02}.png")
        plt.savefig(fpath)
        plt.close()
        print(f'saved as {fpath}')
    return m


files = os.listdir(path)

shipLog = pd.DataFrame([])
for day in range(1, n_day+1):
#for day in range(23, 23+1):
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)

    for target_ship in target_ships:
        print(f'{target_ship} {dt_month:02}{day:02}')
        if target_ship in done_ship:
            print(f'skip {target_ship}')
            continue

        f_name = fr'cur_10minutes{dt_year}{dt_month:02}{day:02}.csv'
        f_path = osp.join(path_ship, target_ship, '2015', f_name)
        path_log = osp.join(path_ship, target_ship, '2015')
        try:
            log = pd.read_csv(f_path, encoding="cp932")
            log['ShipName'] = target_ship
            shipLog = pd.concat([shipLog, log])

            if len(shipLog)==0:
                print(f'Not exist data in {target_ship}, day={day}')
                continue

        except:
            print(f'Error load {f_path}')
            continue

    #save_map_shipLog(shipLog, dt_range, save_path=save_path)
kuroshio_ndata(shipLog)
