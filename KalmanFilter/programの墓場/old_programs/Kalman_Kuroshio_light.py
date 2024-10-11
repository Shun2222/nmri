###
# Kalman Filter 
# フィルタ結果と船の推定偏流との誤差を計算し、船の信頼度を更新する
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

from tqdm import tqdm
from utils import *
from kf_params import *
import logger
import printManager as pm

from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from Ais4ToCurForKalmanTime import ais4
import Ais4ToCurForKalmanTime as atc

pm.clear()

# パラメータの設定
year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'
path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"
save_dir = r"E:\shunsukeE\result\kalman-west-kurosio-Q001"
logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])

pm.printline('Setting jcope loader')
jl = JCOPELoader(year, month)
jl.load_path(path_jcope)

pm.printline('Setting ais loader')
ais_keys = ['cur2', 'lambda2', 'phi2']
ais_outfiles = save_dir + r"\ais_files"
al = atc.AISLoader(year, month, ais_outfiles)
al.set_keys(ais_keys)

pm.printline('Setting Kalman patameter')
# isTarget = pkl.load(open(osp.join('./data/cur_ndata.pkl'), 'rb'))
# _Targets = np.where(isTarget>0)[0]
# enough_data_area = isTarget = pkl.load(open(osp.join('./data/ndataOver600.pkl'), 'rb'))

pm.clear()

for r in range(1):
    # Load jcope and ais 
    pm.printline(f'Loading ais and jcope now (n_slice={r})')
    atc.svm.clear()
    day = 1 
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    dtidx = date_to_dtidx(base_dt, dt)

    data  = al.load_cur(dtidx)
    ais_cur2, ais_lambda2, ais_phi2\
        = [data[key] for key in ais_keys]

    cur2 = kurosio_filter_pooled(ais_cur2, nan_map_pooled, is_pooled=True)# 時刻0のais data
    print(f'ais shape: {ais_cur2.shape}')

    n = _N = _N0 = len(cur2)
    m = _M = 1
    _NM = _N * _M
    _2NM = _2N0 = 2 * _NM

    TF = np.array([True for _ in range(len(cur2))])


    #pkl.dump(TF, open(f'{save_dir}/Targets{year}{month:02}{day:02}-{r}.pkl', 'wb'))
    print(f'Target data num: {_2NM}')

    # Define Kalaman func
    def F_mat(jcope0, jcope1):
        a = 0.5
        F = np.zeros((_2N0+1, _2N0+1))

        for i in range(_2N0):
            F[i][i] = a
            residual = jcope1[i] - a*jcope0[i]
            F[i][-1] = residual 
        F[_2N0][-1] = 1.0
        # print(f'F:{F}F')
        return F

    def H_mat(phi2):
        # min_value = 1/1e10
        H = np.zeros((_N+1, _2NM+1))
        for i in range(_N):
            H[i][i] = np.cos(phi2[i])
            H[i][i+_N] = np.sin(phi2[i])
        H[-1][-1] = 1.0
        return H
        
    def R_mat(sigma2):

        R = np.zeros((_N+1, _N+1))
        for i in range(_N):
            R[i][i] = sigma2[i]
        return R

    def get_z(ais_cur2, dtidx, default=None):

        if len(ais_cur2)==0: 
            print(f'Not exist data({dtidx}) in ais cur 2')
            nan_data = [np.nan for _ in range(_N0)] + [1]
            return np.array(nan_data) 

        ais_cur2_dt = kurosio_filter_pooled(ais_cur2, nan_map_pooled, is_pooled=True)[TF] # 時刻0のais data
        ais_cur12_dt = np.concatenate([ais_cur2_dt, [1]])
        ais_cur12_dt = ais_cur12_dt.reshape(_N0+1, 1)

        tf = np.isnan(ais_cur12_dt)
        if default==None:
            ais_cur12_dt[tf] = np.nan 
        else:
            ais_cur12_dt[tf] = default[tf]
        return ais_cur12_dt

    def get_x(jcope_n, jcope_e, dtidx, default=np.nan):

        jcope_n_dt = kurosio_filter_pooled(jcope_n[dtidx], nan_map_pooled) # 時刻0のjcope data
        jcope_n_dt = jcope_n_dt[TF] # 時刻0のjcope data

        jcope_e_dt = kurosio_filter_pooled(jcope_e[dtidx], nan_map_pooled)[TF] # 時刻0のjcope data

        jcope_ne_dt = np.concatenate([jcope_n_dt, jcope_e_dt])
        jcope_ne_dt = np.concatenate([jcope_ne_dt, [1]])
        jcope_ne_dt = jcope_ne_dt.reshape(_2N0+1, 1)
        jcope_ne_dt[np.isnan(jcope_ne_dt)] = 0.0
        return jcope_ne_dt


    # Setting Kalman Param
    pm.printline('Setting Kalman parameter')

    ## 初期の観測値と状態値
    z = get_z(ais_cur2, dtidx)
    nonZero = z!=0.0
    tf = (~np.isnan(z)) & (nonZero)
    tf_ravel = tf[:_N0].ravel()
    
    lambda2 = kurosio_filter_pooled(ais_lambda2, nan_map_pooled, is_pooled=True)[TF]
    tf_lambda2 = (~np.isnan(lambda2)) & (lambda2>10)
    tf_ravel = (tf_ravel) & (tf_lambda2)
    tf = np.concatenate([tf_ravel, [True]])
    tf = tf.reshape(_N0+1, 1)
    tf2 = np.concatenate([tf_ravel, tf_ravel])
    tf2 = np.concatenate([tf2, [True]])

    phi2 = kurosio_filter_pooled(ais_phi2, nan_map_pooled, is_pooled=True)[TF]

    z = z[tf]
    lambda2 = lambda2[tf_ravel]
    phi2 = phi2[tf_ravel] 
    
    n = _N = int(len(z)-1)
    m = _M = 1 
    _NM = _N * _M
    _2NM = 2 * _NM

    ## H関数
    H = H_mat(phi2)

    ## 初期の推定値x
    jcope_n, jcope_e = jl.load_jcope_day(day)
    x = jcope = get_x(jcope_n, jcope_e, dtidx)
    print(f'jcope shape: {jcope_n[dtidx].shape}')
    assert np.sum(np.isnan(x))==0

    ## F関数
    jcope0 = get_x(jcope_n, jcope_e, dtidx)
    jcope1 = get_x(jcope_n, jcope_e, dtidx+1)
    F = F_mat(jcope0, jcope1)
    ## 分散共分散行列
    Q_value = 0.01
    Q = np.eye(_2N0+1)*Q_value # システム誤差(jcope)
    Q[-1][-1] = 0.0
    logger.record_tabular(f"Q value", Q_value)
    logger.dump_tabular()

    sigma2 = 1/(2*lambda2)
    R = R_mat(sigma2) # 観測誤差(ais)

    P = np.zeros((_2N0+1, _2N0+1)) # 初期事後誤差共分散行列
    I = np.eye(_2NM+1) # 初期事後誤差共分散行列

    ## データの保存
    WHour = (1/2)**(1/8.5)
    pkl.dump(x, open(f'{save_dir}/saverX{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    pkl.dump(jcope, open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    pkl.dump(P, open(f'{save_dir}/saverP{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    pkl.dump(R, open(f'{save_dir}/saverR{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    pkl.dump(F, open(f'{save_dir}/saverF{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    pkl.dump(H, open(f'{save_dir}/saverH{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    jcor = H @ jcope[tf2]
    pkl.dump(jcor, open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
    pkl.dump(H@x[tf2], open(f'{save_dir}/saverXCur{year}{month:02}{day:02}00-{r}.pkl', 'wb'))

    # Filtering開始
    pm.printline('Start KalmanFilter')
    for day in range(1, n_day):
        dt = datetime.datetime(dt_year, dt_month, day)
        print(f'Filter dt:{dt}')
        pm.printline(f'Filtering {dt}')
        
        if day != 1:
            jcope_n, jcope_e = jl.load_jcope_day(day)

        for t in tqdm.tqdm(range(0,24)):
            if day==1 and t==0:
                continue

            dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            data = al.load_cur(dtidx)
            ais_cur2, ais_lambda2, ais_phi2 = [data[key] for key in ais_keys]
            
            if t!=0:
                jcope0 = get_x(jcope_n, jcope_e, dtidx-1)
                jcope1 = get_x(jcope_n, jcope_e, dtidx)
            else:
                jcope0 = get_x(p_jcope_n, p_jcope_e, dtidx-1)
                jcope1 = get_x(jcope_n, jcope_e, dtidx)

            F = F_mat(jcope0, jcope1)

            # 予測ステップ
            pm.printline('Predicting step (update x)')
            #x = x
            x = F @ x #x k|k-1

            pm.printline('Predicting step (update P-)')
            P = F @ P @ F.T + Q
            
            # フィルタリングステップ

            pm.printline('Filtering step (Get z)')
            prev_z = z
            prev_lambda2 = lambda2
            prev_phi2 = phi2

            z = get_z(ais_cur2, dtidx)
            tf = (~np.isnan(z)) & (nonZero)
            tf_ravel = tf[:_N0].ravel()
            
            pm.printline('Filtering step (Get lambda)')
            lambda2 = kurosio_filter_pooled(ais_lambda2, nan_map_pooled, is_pooled=True)[TF]
            tf_lambda2 = (~np.isnan(lambda2)) & (lambda2>10)
            tf_ravel = (tf_ravel) & (tf_lambda2)
            tf = np.concatenate([tf_ravel, [True]])
            tf = tf.reshape(_N0+1, 1)
            tf2 = np.concatenate([tf_ravel, tf_ravel])
            tf2 = np.concatenate([tf2, [True]])

            pm.printline('Filtering step (Get phi)')
            phi2 = kurosio_filter_pooled(ais_phi2, nan_map_pooled, is_pooled=True)[TF]

            z = z[tf]
            z = z.reshape(len(z), 1)
            lambda2 = lambda2[tf_ravel]
            phi2 = phi2[tf_ravel] 
            sigma2 = 1/(2*lambda2)
            
            pm.printline('Filtering step (update parameter)')
            n = _N = int(len(z)-1)
            m = _M = 1 
            _NM = _N * _M
            _2NM = 2 * _NM
            I = np.eye(_2NM+1) 
            print(f'New _N = {_N}')
            
            #kf.R = R_mat(sigma1, sigma2) # 観測誤差(ais)
            pm.printline('Filtering step (calc H)')
            H = H_mat(phi2)

            pm.printline('Filtering step (calc R)')
            R = R_mat(sigma2) # 観測誤差(ais)

            assert np.sum(np.isnan(lambda2))==0
            assert np.sum(np.isnan(sigma2))==0
            assert np.sum(np.isnan(phi2))==0
            assert np.sum(np.isnan(z))==0
            assert np.sum(np.isnan(x))==0
            assert np.sum(np.isnan(P))==0
            assert np.sum(np.isnan(H))==0
            assert np.sum(np.isnan(R))==0


            # kf.update(z)
            pm.printline('Filtering step (calc S)')
            PHT = P[tf2.ravel()].T[tf2.ravel()].T @ H.T
            S = H @ PHT + R
            assert np.sum(np.isnan(S))==0

            inv = 'pinv'
            pm.printline('Filtering step (calc K)')
            SI = np.linalg.pinv(S.T)
            K = PHT @ SI

            pm.printline('Filtering step (filetering x)')
            y = z - H @ x[tf2]
            x[tf2] = x[tf2] + K @ y # x k|k

            pm.printline('Filtering step (update P)')
            I_KH = I - K @ H
            # TODO ちょっと知ってるのと違う，式的にRを写像してPに加えてる，Rの誤差も考慮するようにしてる?
            Ptf = I_KH @ P[tf2.ravel()].T[tf2.ravel()].T @ I_KH.T + K @ R @ K.T
            Ptf_prev = P[tf2.ravel()]
            for i in range(len(Ptf)):
                Ptf_prev[i][tf2.ravel()] = Ptf[i]
            P[tf2.ravel()] = Ptf_prev

            #P = I_KH @ P       
            pm.printline('Calc jcor')
            jcope = get_x(jcope_n, jcope_e, dtidx)
            jcor = H @ jcope[tf2]

            def mean_diff(x, y):
                a = x-y
                diff = np.mean(np.abs(a))
                return diff

            pm.clear()
            logger.record_tabular(f"dtidx", dtidx)
            logger.record_tabular(f"SI", inv)
            logger.record_tabular(f"Available AIS", np.sum(tf_ravel))
            logger.dump_tabular()

            # 保存

            pm.printline('Saving datas')
            pkl.dump(x, open(f'{save_dir}/saverX{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            pkl.dump(z, open(f'{save_dir}/saverZ{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            pkl.dump(jcope, open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            #pkl.dump(P, open(f'{save_dir}/saverP{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            #pkl.dump(R, open(f'{save_dir}/saverR{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            #pkl.dump(F, open(f'{save_dir}/saverF{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            pkl.dump(H, open(f'{save_dir}/saverH{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            #pkl.dump(K, open(f'{save_dir}/saverK{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            pkl.dump(jcor, open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            pkl.dump(H@x[tf2], open(f'{save_dir}/saverXCur{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            pkl.dump(tf, open(f'{save_dir}/saverTarget{year}{month:02}{day:02}{t:02}-{r}.pkl', 'wb'))
            atc.svm.save_info(f'{year}{month:02}{day:02}{t:02}-{r}')

            p_jcope_n = jcope_n
            p_jcope_e = jcope_e
    pm.printline('Finished KalmanFilter')

def avePooling(img,k):
  dst = img.copy()
  w,h = img.shape
  size = k // 2
  for x in range(size, w, k):
    for y in range(size, h, k):
      dst[x-size:x+size,y-size:y+size] = np.mean(img[x-size:x+size,y-size:y+size])
  return dst