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

path_jcope = fr'E:\shunsukeE\data\eas'
path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"

def corr_ais_jcope_ship(ais, jcope, shipLog, target_ship, dt, save=True, save_path=None):
    ais_n = ais[0]
    ais_e = ais[1]
    jcope_n = jcope[0]
    jcope_e = jcope[1]
    #for i in range(len(shipLog)):
    ais_cur_n = []
    ais_cur_e = []
    jcope_cur_n = []
    jcope_cur_e = []

    for i in range(len(shipLog)):
        time = shipLog['DtIdx'].values[i]
        grid0 = shipLog['Grid0'].values[i]
        grid1 = shipLog['Grid1'].values[i]
        curN = shipLog['CurN'].values[i]
        curE = shipLog['CurE'].values[i]
        dtidx = time

        jcope_curN = np.nan
        jcope_curE = np.nan
        if time >= 0:
            if time in jcope_n.keys(): 
                if len(jcope_n[time])!=0: 
                    if grid0<len(jcope_n[dtidx]) and grid1<len(jcope_n[dtidx][0]):
                        jcope_curN = jcope_n[time][grid0][grid1]
            
            if time in jcope_e.keys(): 
                if len(jcope_e[time])!=0:   
                    if grid0<len(jcope_e[dtidx]) and grid1<len(jcope_e[dtidx][0]):
                        jcope_curE = jcope_e[time][grid0][grid1]

        ais_curN = np.nan
        ais_curE = np.nan
        if time in ais_n.keys(): 
            if len(ais_n[time])!=0: 
                if grid0<len(ais_n[dtidx][0]) and grid1<len(ais_n[dtidx][0][0]):
                    if ais_d[time][0][grid0][grid1]>=1e5:
                        ais_curN = ais_n[time][0][grid0][grid1]
        if time in ais_e.keys():
            if len(ais_e[time])!=0: 
                if grid0<len(ais_e[dtidx][0]) and grid1<len(ais_e[dtidx][0][0]):
                    if ais_d[dtidx][0][grid0][grid1]>=1e5:
                        ais_curE = ais_e[time][0][grid0][grid1]
        
        ais_cur_n.append(ais_curN)
        ais_cur_e.append(ais_curE)
        jcope_cur_n.append(jcope_curN)
        jcope_cur_e.append(jcope_curE)

    res = pd.DataFrame([])
    res['DtIdx'] = shipLog['DtIdx'].values

    res['Grid0'] = shipLog['Grid0'].values
    res['Grid1'] = shipLog['Grid1'].values

    res['Ship_CurN'] = shipLog['CurN'].values
    res['Ship_CurE'] = shipLog['CurE'].values

    res['AIS_CurN'] = ais_cur_n
    res['AIS_CurE'] = ais_cur_e

    res['JCOPE_CurN'] = jcope_cur_n
    res['JCOPE_CurE'] = jcope_cur_e
    return res

def save_corr(res, dt, save_path, raw=False):
    year = dt.year 
    month = dt.month

    sns.set(style="whitegrid", palette="muted", color_codes=True)
    plt.rcParams['font.family'] = 'MS Gothic' 
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4*2, 4*2))
    
    ax = axes[0, 0]
    if not raw:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.set_title('Ship-AIS (CurN)')
    res.plot.scatter(x='Ship_CurN', y='AIS_CurN',
            marker='s', c='blue', s=1, alpha=0.5, ax=ax) 

    ax = axes[0, 1]
    if not raw:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.set_title('Ship-JCOPE (CurN)')
    res.plot.scatter(x='Ship_CurN', y='JCOPE_CurN',
            marker='s', c='blue', s=1, alpha=0.5, ax=ax) 

    ax = axes[1, 0]
    if not raw:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.set_title('Ship-AIS (CurE)')
    res.plot.scatter(x='Ship_CurE', y='AIS_CurE',
            marker='s', c='blue', s=1, alpha=0.5, ax=ax) 

    ax = axes[1, 1]
    if not raw:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.set_title('Ship-JCOPE (CurE)')
    res.plot.scatter(x='Ship_CurE', y='JCOPE_CurE',
            marker='s', c='blue', s=1, alpha=0.5, ax=ax) 

    plt.subplots_adjust(wspace=0.3, hspace=0.3) 
    if not raw:
        fname = f'Correlation_Ship-AIS-JCOPE{year}{month}.png'
    else:
        fname = f'Correlation_Ship-AIS-JCOPE{year}{month}-raw.png'

    save_path = osp.join(save_path, fname)
    plt.savefig(save_path)
    plt.close('all')

    print(f'saved {save_path}')

# JCOPE (MAP curN, curE)
jcope_n_path = {}
jcope_e_path = {}
for day in range(1, n_day+1):
    for hour in range(24):
        dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        jcope_n_path[dtidx] = osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "N.csv")
        jcope_e_path[dtidx] = osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "E.csv")
        # jcope_n[dtidx] = pd.read_csv(osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "N.csv"), encoding="cp932", header=None)
        # jcope_n[dtidx] = jcope_n[dtidx].values
        # jcope_e[dtidx] = pd.read_csv(osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "E.csv"), encoding="cp932", header=None)
        # jcope_e[dtidx] = jcope_e[dtidx].values

ais_n_path = {}
ais_e_path = {}
ais_d_path = {}
patterns = [f'(\w+){month:02}..[0-9][0-9]N.csv', 
            f'(\w+){month:02}..[0-9][0-9]E.csv', 
            f'(\w+){month:02}..[0-9][0-9]D.csv']
files = os.listdir(path_ais)
dirs = [f for f in files if os.path.isdir(path_ais)]
for d in dirs:
    path2 = osp.join(path_ais, d, 'log')
    files = os.listdir(path2)
    filenames = [f for f in files if os.path.isfile(osp.join(path2, f))]
    for f in filenames:
        if re.match(patterns[0], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_n_path.keys():
                ais_n_path[dtidx] = []
                ais_n_path[dtidx].append(f_path)
                # ais_n_path[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_n_path[dtidx][-1] = ais_n_path[dtidx][-1].values
            else:
                ais_n_path[dtidx].append(f_path)
                # ais_n[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_n[dtidx][-1] = ais_n[dtidx][-1].values         
        if re.match(patterns[1], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_e_path.keys():
                ais_e_path[dtidx] = []
                ais_e_path[dtidx].append(f_path)
            else:
                ais_e_path[dtidx].append(f_path)

        if re.match(patterns[2], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_d_path.keys():
                ais_d_path[dtidx] = []
                ais_d_path[dtidx].append(f_path)
            else:
                ais_d_path[dtidx].append(f_path)    

files = os.listdir(path)
target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]

ress = {}
for day in range(1, n_day+1):
#for day in range(23, 23+1):
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    print(dt)
    def read_ais(year, month, day):
        ais_n = {}
        ais_e = {}
        ais_d = {}

        for hour in range(24):
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)

            dtidx = date_to_dtidx(base_dt, dt)
            ais_n[dtidx] = []
            ais_e[dtidx] = []
            ais_d[dtidx] = []
            
            if dtidx in ais_n_path.keys():
                for i in range(len(ais_n_path[dtidx])):
                    try:
                        ais_n[dtidx].append(pd.read_csv(ais_n_path[dtidx][i], encoding="cp932", header=None))
                        ais_n[dtidx][-1] = ais_n[dtidx][-1].values
                    except:
                        print(f'Error load {ais_n_path[dtidx][i]}')
            
            if dtidx in ais_e_path.keys():
                for i in range(len(ais_e_path[dtidx])):
                    try:
                        ais_e[dtidx].append(pd.read_csv(ais_e_path[dtidx][i], encoding="cp932", header=None))
                        ais_e[dtidx][-1] = ais_e[dtidx][-1].values  
                    except:
                        print(f'Error load {ais_e_path[dtidx][i]}')            
            
            if dtidx in ais_d_path.keys():
                for i in range(len(ais_d_path[dtidx])):
                    try:
                        ais_d[dtidx].append(pd.read_csv(ais_d_path[dtidx][i], encoding="cp932", header=None))
                        ais_d[dtidx][-1] = ais_d[dtidx][-1].values  
                    except:
                        print(f'Error load {ais_d_path[dtidx][i]}')

        return ais_n, ais_e, ais_d

    def read_jcope(year, month, day):
        jcope_n = {}
        jcope_e = {}

        for hour in range(24):
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            jcope_n[dtidx] = []
            jcope_e[dtidx] = []
            try:
                jcope_n[dtidx] = pd.read_csv(jcope_n_path[dtidx], encoding="cp932", header=None)
                jcope_n[dtidx] = jcope_n[dtidx].values
            except:
                print(f'Error load {jcope_n_path[dtidx]}')

            try:
                jcope_e[dtidx] = pd.read_csv(jcope_e_path[dtidx], encoding="cp932", header=None)
                jcope_e[dtidx] = jcope_e[dtidx].values
            except:
                print(f'Error load {jcope_e_path[dtidx]}')

        return jcope_n, jcope_e
    
    ais_n, ais_e, ais_d = read_ais(dt_year, dt_month, day)
    jcope_n, jcope_e = read_jcope(dt_year, dt_month, day)


    for target_ship in target_ships:
        print(f'{target_ship} {dt_month:02}{day:02}')
        if target_ship in done_ship:
            print(f'skip {target_ship}')
            continue

        f_name = fr'cur_10minutes{dt_year}{dt_month:02}{day:02}.csv'
        f_path = osp.join(path_ship, target_ship, '2015', f_name)
        path_log = osp.join(path_ship, target_ship, '2015')
        try:
            shipLog = pd.read_csv(f_path, encoding="cp932")
            shipLog['ShipName'] = target_ship

            if len(shipLog)==0:
                print(f'not exist data in {target_ship}, day={day}')
                continue

        except:
            print(f'Error load {f_path}')
            continue

        print(f'ais n {len(ais_n)}, ais e {len(ais_e)}, ais d {len(ais_d)}')
        print(f'jcope n {len(jcope_n)}, jcope e {len(jcope_e)}')
        print(f'shiplog {len(shipLog)}')

        ais = [ais_n, ais_e]
        jcope = [jcope_n, jcope_e]

        save_path = osp.join(path_ship, target_ship, '2015')
        res = corr_ais_jcope_ship(ais, jcope, shipLog, target_ship, dt, save=True, save_path=save_path)
        if not target_ship in ress.keys():
            ress[target_ship] = res
        else:
            ress[target_ship] = pd.concat([ress[target_ship], res])

ress2 = pd.DataFrame([]) 
for target_ship in target_ships:
    if target_ship in ress.keys():
        save_path = osp.join(path_ship, target_ship, '2015')
        save_corr(ress[target_ship], dt, save_path)
        save_corr(ress[target_ship], dt, save_path, raw=True)
        ress2 = pd.concat([ress2, ress[target_ship]])
save_path = path_ship
save_corr(ress2, dt, save_path)
save_corr(ress2, dt, save_path, raw=True)


    