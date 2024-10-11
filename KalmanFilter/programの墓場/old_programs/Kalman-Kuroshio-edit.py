###
# Kalman Filter 
###
print('\rimport files now', end='')
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
import pickle as pkl
import math
import os 
import os.path as osp
import re
import pickle
from filterpy.common import Saver
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from tqdm import tqdm
from utils import *
from kf_params import *
import logger
import printManager as pm

pm.clear()

#パラメータの設定
year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'
path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"
save_dir = r"E:\shunsukeE\result\kalman-Q0001"
logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])


jl = JCOPELoader(year, month)
jl.load_path(path_jcope)

ais_keys = ['cur1', 'cur2', 'lambda1', 'lambda2', 'phi1', 'phi2']
al = AISLoader(year, month)
al.load_path(keys=ais_keys)


day = 1
data = al.load_ais_day(day)
ais_cur1 = data['cur1']
jcope_n, jcope_e = jl.load_jcope_day(day)

dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
dtidx = date_to_dtidx(base_dt, dt)
cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map)# 時刻0のais data

_Nslice = 20
_2Nslice = _Nslice * 2 
#isTarget = pkl.load(open(osp.join('./data/cur_ndata.pkl'), 'rb'))
#_Targets = np.where(isTarget>0)[0]
#enough_data_area = isTarget = pkl.load(open(osp.join('./data/ndataOver600.pkl'), 'rb'))
isExistShipLog = isTarget = pkl.load(open(osp.join('./data/isExistShipLog.pkl'), 'rb'))
_Targets = np.where(isTarget)[0]
logger.record_tabular(f"SliceNum", _Nslice)
logger.record_tabular(f"TargetNum", len(_Targets))
logger.dump_tabular()
with open(f'{save_dir}/Targets{year}{month:02}.pkl', 'wb') as f:
    pickle.dump(isTarget, f)
#for r in range(len(cur1)//_Nslice + 1):
for r in range(len(_Targets)//_Nslice+1):
# Load jcope and ais 
    pm.printline('Loading ais and jcope now')
    day = 1
    data  = al.load_ais_day(day)
    ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2\
        = [data[key] for key in ais_keys]

    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    dtidx = date_to_dtidx(base_dt, dt)
    cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map)# 時刻0のais data

    n = _N = _Nslice
    m = _M = 1
    _NM = _N * _M
    _2NM = 2 * _NM

    TF = np.array([False for _ in range(len(cur1))])
    targets = _Targets[r*_N:(r+1)*_N]
    for target in targets:
        print(target)
        TF[target] = True
    if np.sum(TF)!=_Nslice:
        _N = np.sum(TF)
        m = _M = 1
        _NM = _N * _M
        _2NM = 2 * _NM
    with open(f'{save_dir}/Targets{year}{month:02}{day:02}-{r}.pkl', 'wb') as f:
        pickle.dump(TF, f)
    print(f'Target data num: {_2NM}')
    print(f'Target exist data num: {isTarget[TF]}')

# Define Kalaman func
    def F_mat(jcope0, jcope1):
        a = 0.5
        F = np.zeros((_2Nslice+1, _2Nslice+1))

        for i in range(_2Nslice):
            F[i][i] = a
            residual = jcope1[i] - a*jcope0[i]
            F[i][-1] = residual 
        F[_2Nslice][-1] = 1.0
        #print(f'F:{F}F')
        return F

# TODO
    def H_mat(phi1, phi2):

        #min_value = 1/1e10
        H = np.zeros((_2NM+1, _2NM+1))
        for i in range(_N):
            H[i][i] = np.cos(phi1[i])
            H[i][i+_N] = np.sin(phi1[i])
            H[i+_N][i] = np.cos(phi2[i])
            H[i+_N][i+_N] = np.sin(phi2[i])
            #H[i][i] = np.cos(phi1[i]+np.pi)
            # H[i][i+_N] = np.sin(phi1[i]+np.pi)
            # H[i+_N][i] = np.cos(phi1[i]+np.pi*3/2)
            # H[i+_N][i+_N] = np.sin(phi1[i]+np.pi*3/2)

        H[-1][-1] = 1.0
        return H

            
        
    def R_mat(sigma1, sigma2):

        R = np.zeros((_2NM+1, _2NM+1))
        for i in range(_N):
            R[i][i] = sigma1[i]
            R[i+_N][i+_N] = sigma2[i]
            #R[i][i] = 0.001 
            #R[i+_N][i+_N] = 0.001 
        #print(f'R:{R}R')
        return R

    def get_z(ais_cur1, ais_cur2, dtidx, default=None):

        if not dtidx in ais_cur1.keys() or not dtidx in ais_cur2.keys():
            print(f'Not exist key({dtidx}) in ais cur 1 or 2')
            nan_data = [np.nan for _ in range(_2Nslice)] + [1]
            return np.array(nan_data) 
        if len(ais_cur1[dtidx])==0 or len(ais_cur2[dtidx])==0: 
            print(f'Not exist data({dtidx}) in ais cur 1 or 2')
            nan_data = [np.nan for _ in range(_2Nslice)] + [1]
            return np.array(nan_data) 

        ais_cur1_dt = kurosio_filter(ais_cur1[dtidx][0], nan_map)[TF] # 時刻0のais data
        ais_cur2_dt = kurosio_filter(ais_cur2[dtidx][0], nan_map)[TF] # 時刻0のais data

        ais_cur12_dt = np.concatenate([ais_cur1_dt, ais_cur2_dt])
        ais_cur12_dt = np.concatenate([ais_cur12_dt, [1]])
        ais_cur12_dt = ais_cur12_dt.reshape(_2Nslice+1, 1)

        tf = np.isnan(ais_cur12_dt)
        if default==None:
            ais_cur12_dt[tf] = np.nan 
        else:
            ais_cur12_dt[tf] = default[tf]
        return ais_cur12_dt

    def get_x(jcope_n, jcope_e, dtidx, default=np.nan):

        jcope_n_dt = kurosio_filter(jcope_n[dtidx], nan_map)[TF] # 時刻0のjcope data
        jcope_e_dt = kurosio_filter(jcope_e[dtidx], nan_map)[TF] # 時刻0のjcope data

        jcope_ne_dt = np.concatenate([jcope_n_dt, jcope_e_dt])
        jcope_ne_dt = np.concatenate([jcope_ne_dt, [1]])
        jcope_ne_dt = jcope_ne_dt.reshape(_2Nslice+1, 1)
        jcope_ne_dt[np.isnan(jcope_ne_dt)] = 0.0
        return jcope_ne_dt


# Setting Kalman Param
    pm.printline('Setting Kalman param now')

## 初期の観測値と状態値
    z = get_z(ais_cur1, ais_cur2, dtidx)
    nonZero = z!=0.0
    nonZero = nonZero[:_Nslice] & nonZero[_Nslice:-1]
    nonZero = np.concatenate([nonZero, nonZero])
    nonZero = np.concatenate([nonZero, [[True]]])
    tf = (~np.isnan(z)) & (nonZero)
    tf_ravel = tf.ravel()
    tf_ravel_hl = tf_ravel[:_Nslice]
    z = z[tf]

    n = _N = int(len(z)/2)
    m = _M = 1 
    _NM = _N * _M
    _2NM = 2 * _NM

    x = jcope = get_x(jcope_n, jcope_e, dtidx)
    assert np.sum(np.isnan(x))==0

## F関数
#kf.F = F_mat(jcope_n, jcope_e, dtidx+1)
    jcope0 = get_x(jcope_n, jcope_e, dtidx)
    jcope1 = get_x(jcope_n, jcope_e, dtidx+1)
    F = F_mat(jcope0, jcope1)

## H関数
    lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)[TF]
    lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)[TF]
    lambda1 = lambda1[tf_ravel_hl]
    lambda2 = lambda2[tf_ravel_hl]
    lambda1[lambda1==0] = 0.001
    lambda2[lambda2==0] = 0.001

    phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)[TF]
    phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)[TF]
    phi1 = phi1[tf_ravel_hl]
    phi2 = phi2[tf_ravel_hl]

    H = H_mat(phi1, phi2)

## 分散共分散行列
    Q = np.eye(_2Nslice+1)*0.001 # システム誤差(jcope)

    sigma1 = 1/(2*lambda1)
    sigma2 = 1/(2*lambda2)
    R = R_mat(sigma1, sigma2) # 観測誤差(ais)

    P = np.zeros((_2Nslice+1, _2Nslice+1)) # 初期事後誤差共分散行列
    I = np.eye(_2NM+1) # 初期事後誤差共分散行列

## logの記録

    WHour = (1/2)**(1/8.5)

    with open(f'{save_dir}/saverX{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(x, f)
    with open(f'{save_dir}/saverZ{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(z, f)
    with open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(jcope, f)

    with open(f'{save_dir}/saverP{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(P, f)
    with open(f'{save_dir}/saverR{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(R, f)
    with open(f'{save_dir}/saverF{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(F, f)
    with open(f'{save_dir}/saverH{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(H, f)
    jcor = H @ jcope[tf]
    with open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(jcor, f)
    with open(f'{save_dir}/saverXCur{year}{month:02}{day:02}00-{r}.pkl', 'wb') as f:
        pickle.dump(H@x[tf], f)

# Filtering開始
    pm.printline('Start KalmanFilter')
    for day in range(1, n_day):
        dt = datetime.datetime(dt_year, dt_month, day)
        print(f'Filter dt:{dt}')
        pm.printline(f'Filtering {dt}')

        data = al.load_ais_day(day)
        ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2 = [data[key] for key in ais_keys]
        
        jcope_n, jcope_e = jl.load_jcope_day(day)

        for t in tqdm.tqdm(range(1,23)):
            dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if 0<t and t<24:
                jcope0 = get_x(jcope_n, jcope_e, dtidx-1)
                jcope1 = get_x(jcope_n, jcope_e, dtidx)
                F = F_mat(jcope0, jcope1)

            # 予測ステップ
            pm.printline('Predicting now')
            #kf.predict()

            #x = x
            x = F @ x #x k|k-1
            P = F @ P @ F.T + Q
            
            # フィルタリングステップ
            pm.printline('Filtering now')

            prev_z = z
            prev_lambda1 = lambda1
            prev_lambda2 = lambda2
            prev_phi1 = phi1
            prev_phi2 = phi2

            z = get_z(ais_cur1, ais_cur2, dtidx)
            nonZero = z!=0.0
            nonZero = nonZero[:_Nslice] & nonZero[_Nslice:-1]
            nonZero = np.concatenate([nonZero, nonZero])
            nonZero = np.concatenate([nonZero, [[True]]])
            tf = (~np.isnan(z)) & (nonZero)
            tf_ravel = tf.ravel()
            tf_ravel_hl = tf_ravel[:_Nslice]
            #z[tf]= prev_z[tf]
            #print(f'z: {z}')
            #print(f'tf: {tf}')
            z = z[tf]
            n = _N = int(len(z)/2)
            m = _M = 1 
            _NM = _N * _M
            _2NM = 2 * _NM
            I = np.eye(_2NM+1) 
            print(f'New _N = {_N}')

            lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)[TF] 
            lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)[TF]

            lambda1 = lambda1[tf_ravel_hl]
            lambda2 = lambda2[tf_ravel_hl]

            lambda1[lambda1==0] = 0.001
            lambda2[lambda2==0] = 0.001

            isNan = np.isnan(lambda1) 
            lambda1[isNan] = 0.001

            isNan = np.isnan(lambda2) 
            lambda2[isNan] = 0.001

            phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)[TF]
            phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)[TF]
            phi1 = phi1[tf_ravel_hl] 
            phi2 = phi2[tf_ravel_hl] 

            sigma1 = 1/(2*lambda1)
            sigma2 = 1/(2*lambda2)

            #kf.R = R_mat(sigma1, sigma2) # 観測誤差(ais)
            H = H_mat(phi1, phi2)
            R = R_mat(sigma1, sigma2) # 観測誤差(ais)

            assert np.sum(np.isnan(lambda1))==0
            assert np.sum(np.isnan(lambda2))==0
            assert np.sum(np.isnan(sigma1))==0
            assert np.sum(np.isnan(sigma2))==0
            assert np.sum(np.isnan(phi1))==0
            assert np.sum(np.isnan(phi2))==0
            assert np.sum(np.isnan(z))==0
            assert np.sum(np.isnan(x))==0
            assert np.sum(np.isnan(P))==0
            assert np.sum(np.isnan(H))==0
            assert np.sum(np.isnan(R))==0


            #kf.update(z)
            PHT = P[tf_ravel].T[tf_ravel].T @ H.T
            S = H @ PHT + R
            assert np.sum(np.isnan(S))==0

            inv = 'pinv'
            SI = np.linalg.pinv(S.T)
            K = PHT @ SI

            y = z - H @ x[tf]
            x[tf] = x[tf] + K @ y # x k|k

            I_KH = I - K @ H
            # TODO ちょっと知ってるのと違う，式的にRを写像してPに加えてる，Rの誤差も考慮するようにしてる?
            Ptf = I_KH @ P[tf_ravel].T[tf_ravel].T @ I_KH.T + K @ R @ K.T
            Ptf_prev = P[tf_ravel]
            for i in range(len(Ptf)):
                Ptf_prev[i][tf_ravel] = Ptf[i]
            P[tf_ravel] = Ptf_prev

            #P = I_KH @ P         
            jcope = get_x(jcope_n, jcope_e, dtidx)
            jcor = H @ jcope[tf]
                   
                

            # 関数の更新
            pm.printline('Updating function now')
            #kf.F = F_mat(jcope_n, jcope_e, dtidx+1)
            #kf.H = H_mat(lambda1, lambda2, phi1, phi2)

            def mean_diff(x, y):
                a = x-y
                diff = np.mean(np.abs(a))
                return diff

            pm.clear()
            logger.record_tabular(f"dtidx", dtidx)
            logger.record_tabular(f"slice num", r)
            logger.record_tabular(f"SI", inv)
            logger.record_tabular(f"AIS-JCOPE", mean_diff(z, H@jcope[tf]))
            logger.record_tabular(f"Kalman-JCOPE", mean_diff(H@x[tf], H@jcope[tf]))
            logger.record_tabular(f"AIS-Kalman", mean_diff(z, H@x[tf]))
            logger.record_tabular(f"Available AIS", np.sum(tf_ravel))
            logger.dump_tabular()

            # 保存
            with open(f'{save_dir}/saverX{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(x, f) #KALMAN
            with open(f'{save_dir}/saverZ{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(z, f) #AIS
            with open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(jcope, f) #JCOPE
            with open(f'{save_dir}/saverP{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(P, f)
            with open(f'{save_dir}/saverR{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(R, f)
            with open(f'{save_dir}/saverF{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(F, f)
            with open(f'{save_dir}/saverH{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(H, f)
            with open(f'{save_dir}/saverK{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(K, f)
            with open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(jcor, f)
            with open(f'{save_dir}/saverXCur{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(H@x[tf], f)
            with open(f'{save_dir}/saverTarget{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb') as f:
                pickle.dump(tf, f)
    pm.printline('Finished KalmanFilter')

    
