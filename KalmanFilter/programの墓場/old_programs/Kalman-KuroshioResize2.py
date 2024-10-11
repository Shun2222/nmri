###
# Kalman Filter 
###
print('\rimport files now', end='')
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import randn
import seaborn as sns
import pickle as pkl
import math
import os 
import os.path as osp
import re
import pickle

from tqdm import tqdm
from utils import *
from kf_params import *
import logger
import printManager as pm
from ais_loader import AISLoader
from jcope_loader import JCOPELoader
from my_kalman_filter import KalmanFilter

pm.clear()

#パラメータの設定
year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = 3 #nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'
#path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"
save_dir = r"E:\shunsukeE\result\kalman-test"
logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])


jl = JCOPELoader(year, month)
jl.load_path(path_jcope)

ais_keys = ['cur1', 'cur2', 'lambda1', 'lambda2', 'phi1', 'phi2']
al = AISLoader(year, month)
al.load_path(keys=ais_keys)

# Define Kalaman func
def F_mat(jcope_n, jcope_e, dtidx, isTarget):
    a = 0.5
    jcope0 = get_x(jcope_n, jcope_e, dtidx-1, isTarget)
    jcope1 = get_x(jcope_n, jcope_e, dtidx, isTarget)

    F = np.zeros((len(jcope0)+1, len(jcope0)+1))

    for i in range(len(jcope0)):
        F[i][i] = a
        residual = jcope1[i] - a*jcope0[i]
        F[i][-1] = residual 
    F[_N][-1] = 1.0
    #print(f'F:{F}F')
    return F

# TODO
def H_mat(ais_phi1, ais_phi2, dtidx, isTarget, notNan):
    if np.sum(notNan)==0:
        return np.zeros((1, 1))
    phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)[isTarget]
    phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)[isTarget]
    phi1 = phi1[notNan]
    phi2 = phi2[notNan]

    H = np.zeros((len(phi1)*2+1, len(phi1)*2+1))
    for i in range(_NHalf):
        H[i][i] = np.cos(phi1[i])
        H[i][i+_NHalf] = np.sin(phi1[i])
        H[i+_NHalf][i] = np.cos(phi2[i])
        H[i+_NHalf][i+_NHalf] = np.sin(phi2[i])
    H[-1][-1] = 1.0
    return H

        
    
def R_mat(ais_lambda1, ais_lambda2, dtidx, isTarget, notNan):
    if np.sum(notNan)==0:
        return np.zeros((1, 1))
    lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)[isTarget]
    lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)[isTarget]
    lambda1 = lambda1[notNan]
    lambda2 = lambda2[notNan]
    sigma1 = 1/(2*lambda1)
    sigma2 = 1/(2*lambda2)

    R = np.zeros((len(sigma1)*2+1, len(sigma1)*2+1))
    for i in range(_NHalf):
        R[i][i] = sigma1[i]
        R[i+_NHalf][i+_NHalf] = sigma2[i]
    return R

def get_z(ais_cur1, ais_cur2, dtidx, isTarget):

    if not dtidx in ais_cur1.keys() or not dtidx in ais_cur2.keys():
        return np.array([]) 
    min_value = 0.0 #1/1e10
    ais_cur1_dt = kurosio_filter(ais_cur1[dtidx][0], nan_map)[isTarget] # 時刻0のais data
    ais_cur2_dt = kurosio_filter(ais_cur2[dtidx][0], nan_map)[isTarget] # 時刻0のais data

    ais_cur12_dt = np.concatenate([ais_cur1_dt, ais_cur2_dt])
    ais_cur12_dt = np.concatenate([ais_cur12_dt, [1]])
    ais_cur12_dt = ais_cur12_dt.reshape(len(ais_cur12_dt), 1)

    return ais_cur12_dt

def get_x(jcope_n, jcope_e, dtidx, isTarget):

    jcope_n_dt = kurosio_filter(jcope_n[dtidx], nan_map)[isTarget] # 時刻0のjcope data
    jcope_e_dt = kurosio_filter(jcope_e[dtidx], nan_map)[isTarget] # 時刻0のjcope data

    jcope_ne_dt = np.concatenate([jcope_n_dt, jcope_e_dt])
    jcope_ne_dt = np.concatenate([jcope_ne_dt, [1]])
    jcope_ne_dt = jcope_ne_dt.reshape(len(jcope_ne_dt), 1)
    return jcope_ne_dt

day = 1
data = al.load_ais_day(day)
ais_cur1 = data['cur1']
jcope_n, jcope_e = jl.load_jcope_day(day)

dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
dtidx = date_to_dtidx(base_dt, dt)
cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map)# 時刻0のais data
_Ncur = len(cur1)

_NHalf = 20
_M = 1
_N = 2 * _NHalf + 1
for r in range(_Ncur//_NHalf + 1):
# Load jcope and ais 
    pm.printline('Loading ais and jcope now')
    day = 1
    data  = al.load_ais_day(day)
    ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2\
        = [data[key] for key in ais_keys]

    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    dtidx = date_to_dtidx(base_dt, dt)
    cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map)# 時刻0のais data

    isTarget = np.array([False for _ in range(len(cur1))])
    isTarget[r*_NHalf:(r+1)*_NHalf] = True
    isTarget[r*_NHalf+len(cur1)*2:(r+1)*_NHalf+len(cur1)*2] = True

# Setting Kalman Param
    pm.printline('Setting Kalman param now')

## 初期の観測値と状態値
    z = get_z(ais_cur1, ais_cur2, dtidx, isTarget)
    notNan = ~np.isnan(z)
    notNan_ravel = notNan.ravel()
    notNan_ravel_hl = notNan_ravel[:_NHalf]
    z = z[notNan]

    x = jcope = get_x(jcope_n, jcope_e, dtidx, isTarget)
    assert np.sum(np.isnan(x))==0

## F関数
    F = F_mat(jcope_n, jcope_e, dtidx+1, isTarget)

## H関数
    H = H_mat(ais_phi1, ais_phi2, dtidx, isTarget, notNan_ravel_hl)

## 分散共分散行列
    Q = np.eye(_N)*0.001 # システム誤差(jcope)
    R = R_mat(ais_lambda1, ais_lambda2, dtidx, isTarget, notNan_ravel_hl) # 観測誤差(ais)

    kf = KalmanFilter(logger, _N, _M, x, z, notNan, F, H, Q, R)
    fname = f'{dt_year}{dt_month:02}{day:02}00-{r*_N}'
    kf.save(jcope[notNan], fname)
# Filtering開始
    pm.printline('Start KalmanFilter')
    for day in range(1, n_day):
        dt = datetime.datetime(dt_year, dt_month, day)
        print(f'Filter dt:{dt}')
        pm.printline(f'Filtering {dt}')

        ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2,_,_,_\
            = al.load_ais_day(day)
        
        jcope_n, jcope_e = jl.load_jcope_day(day)

        for t in tqdm.tqdm(range(1,23)):
            dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if 0<t and t<24:
                F = F_mat(jcope_n, jcope_e, dtidx, isTarget)

            # 予測ステップ
            pm.printline('Predicting now')
            kf.predict(F, Q)
            
            # フィルタリングステップ
            pm.printline('Filtering now')

            z = get_z(ais_cur1, ais_cur2, dtidx, isTarget)
            notNan = ~np.isnan(z)
            notNan_ravel = notNan.ravel()
            notNan_ravel_hl = notNan_ravel[:_NHalf]
            z = z[notNan]
            I = np.eye(len(z)+1) 
            print(f'New _NHalf = {_NHalf}')

            H = H_mat(ais_phi1, ais_phi2, dtidx, isTarget, notNan_ravel_hl)
            R = R_mat(ais_lambda1, ais_lambda2, dtidx, isTarget, notNan_ravel_hl)

            assert np.sum(np.isnan(z))==0
            assert np.sum(np.isnan(x))==0
            assert np.sum(np.isnan(P))==0
            assert np.sum(np.isnan(H))==0
            assert np.sum(np.isnan(R))==0

            kf.update(z)

            pm.clear()
            jcope = get_x(jcope_n, jcope_e, dtidx, isTarget)
            kf.logger(jcope)
            fname = f'{dt_year}{dt_month:02}{day:02}{t:02}-{r*_N}'
            kf.save(jcope, fname)

    pm.printline('Finished KalmanFilter')

    
