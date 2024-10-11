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
import pickle
from filterpy.common import Saver
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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
files = os.listdir(path)
target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]

ress = {}
for day in range(1, n_day+1):
#for day in range(23, 23+1):
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    print(dt)
    ais_n, ais_e, ais_d = read_ais(dt_year, dt_month, day)
    jcope_n, jcope_e = read_jcope(dt_year, dt_month, day)

    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    dtidx = date_to_dtidx(base_dt, dt)
    data = kurosio_filter(ais_n[dtidx][0], nan_map)
    n = len(data)
    m = 1

    def F_element(i, j, k, data):
        a = 0.5
        if i==j:
            return a
        elif i==n+m:
            jcope0 = data[k-1]
            jcope1 = data[k]
            residual = jcope1[i] - jcope0[i]
            return residual
        else:
            return 0

    # TODO
    def H_element(i, j):
        if i==j:
            return 1
        else:
            return 0
        
    def R_mat(sigma):
        R = np.zeros((n*m+1, n*m+1))
        for i in range(len(sigma)):
            R[i][i] = sigma[i]
        return R


    kf = KalmanFilter(dim_x=n*m+1, dim_z=n*m+1)
    ais_n0 = kurosio_filter(ais_n[dtidx][0], nan_map) # 時刻0のais data
    ais_n0 = np.concatenate([ais_n0, [1]])
    ais_n0 = ais_n0.reshape(len(ais_n0), 1)
    ais_n0[ais_n0!=ais_n0] = 0

    jcope_n0 = kurosio_filter(jcope_n[dtidx], nan_map) # 時刻0のjcope data
    jcope_n0 = np.concatenate([jcope_n0, [1]])
    jcope_n0 = jcope_n0.reshape(len(jcope_n0), 1)
    jcope_n0[jcope_n0!=jcope_n0] = 0

    # F関数, H関数
    kf.F = np.array([[F_element(i, j, 1, jcope_n0)  for i in range(n*m+1)] for j in range(n*m+1)])
    kf.H = np.array([[H_element(i, j)  for i in range(n*m+1)] for j in range(n*m+1)])

    # 分散共分散行列
    kf.Q = np.eye(n*m+1)*0.001 # システム誤差(jcope)

    sigma = 1/kurosio_filter(ais_d[dtidx][0], nan_map)
    kf.R = R_mat(sigma) # 観測誤差(ais)
    #kf.R = np.eye(n*m+1)*0.001

    kf.x = jcope_n0 # 初期位置
    kf.P = np.eye(n*m+1) # 初期事後誤差共分散行列

    print('Start KalmanFilter')
    for t in tqdm.tqdm(range(1,24)):

        # 予測ステップ
        kf.predict()
        
        # フィルタリングステップ
        dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        sigma = 1/kurosio_filter(ais_d[dtidx][0], nan_map)
        kf.R = R_mat(sigma) # 観測誤差(ais)
        
        y = kurosio_filter(ais_n[dtidx][0], nan_map)
        y = np.concatenate([y, [1]])
        y = y.reshape(len(y), 1)
        y[y!=y] = 0
        # nanの処理と誤差の最大化
        kf.update(y)
        
        with open(f'{save_dir}/saverX{year}{month:02}{t:02}.pkl', 'wb') as f:
            pickle.dump(kf.x, f)
        with open(f'{save_dir}/saverZ{year}{month:02}{t:02}.pkl', 'wb') as f:
            pickle.dump(kf.z, f)
            
        # 関数の更新
        x = kurosio_filter(jcope_n[t], nan_map)
        x = np.concatenate([x, [1]])
        x = x.reshape(len(x), 1)
        x[x!=x] = 0
        kf.F = np.array([[F_element(i, j, t, x)  for i in range(n*m+1)] for j in range(n*m+1)])
        kf.H = np.array([[H_element(i, j)  for i in range(n*m+1)] for j in range(n*m+1)])

    print('Finished KalmanFilter')

    