###
# x軸を各船の偏流，y軸をAISもしくはJCOPEの偏流として図を作成する
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
n_day = 2 #nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'
path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"
save_dir = r"E:\shunsukeE\result\filtering-test"
logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])


# JCOPEのファイルパスの読み取り
print('\rReading jcope path now', end='')
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
assert len(jcope_n_path)!=0
assert len(jcope_e_path)!=0

# AISのファイルパスの読み取り
pm.printline('Reading ais path now')
ais_cur1_path = {} #偏流1
ais_cur2_path = {} #偏流2
ais_lambda1_path = {} #固有値1
ais_lambda2_path = {} #固有値2
ais_phi1_path = {} #固有ベクトルの方向1
ais_phi2_path = {} #固有ベクトルの方向2

patterns = [f'(\w+){month:02}..[0-9][0-9]Cur1.csv', 
            f'(\w+){month:02}..[0-9][0-9]Cur2.csv', 
            f'(\w+){month:02}..[0-9][0-9]Lambda1.csv',
            f'(\w+){month:02}..[0-9][0-9]Lambda2.csv',
            f'(\w+){month:02}..[0-9][0-9]Phi1.csv',
            f'(\w+){month:02}..[0-9][0-9]Phi2.csv']

dirs = []
for day in range(1, n_day+1):
    for hour in range(1, 13):
        d = osp.join(f'{year-int(year/100)*100}{month:02}{day:02}-{hour}log', 'log')
        dirs.append(d)
for d in dirs:
    path2 = osp.join(path_ais, d)
    files = os.listdir(path2)
    filenames = [f for f in files if os.path.isfile(osp.join(path2, f))]
    for f in filenames:
        if re.match(patterns[0], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_cur1_path.keys():
                ais_cur1_path[dtidx] = []
                ais_cur1_path[dtidx].append(f_path)
                # ais_cur1_path[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_cur1_path[dtidx][-1] = ais_cur1_path[dtidx][-1].values
            else:
                ais_cur1_path[dtidx].append(f_path)
                # ais_cur1[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_cur1[dtidx][-1] = ais_cur1[dtidx][-1].values         

        if re.match(patterns[1], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_cur2_path.keys():
                ais_cur2_path[dtidx] = []
                ais_cur2_path[dtidx].append(f_path)
                # ais_cur2_path[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_cur2_path[dtidx][-1] = ais_cur2_path[dtidx][-1].values
            else:
                ais_cur2_path[dtidx].append(f_path)
                # ais_cur2[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_cur2[dtidx][-1] = ais_cur2[dtidx][-1].values         


        if re.match(patterns[2], f):
            f_path = osp.join(path2, f)
            day = int(f[-15:-13])
            hour = int(f[-13:-11])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_lambda1_path.keys():
                ais_lambda1_path[dtidx] = []
                ais_lambda1_path[dtidx].append(f_path)
            else:
                ais_lambda1_path[dtidx].append(f_path)    

        if re.match(patterns[3], f):
            f_path = osp.join(path2, f)
            day = int(f[-15:-13])
            hour = int(f[-13:-11])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_lambda2_path.keys():
                ais_lambda2_path[dtidx] = []
                ais_lambda2_path[dtidx].append(f_path)
            else:
                ais_lambda2_path[dtidx].append(f_path)    

        if re.match(patterns[4], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_phi1_path.keys():
                ais_phi1_path[dtidx] = []
                ais_phi1_path[dtidx].append(f_path)
            else:
                ais_phi1_path[dtidx].append(f_path)    

        if re.match(patterns[5], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_phi2_path.keys():
                ais_phi2_path[dtidx] = []
                ais_phi2_path[dtidx].append(f_path)
            else:
                ais_phi2_path[dtidx].append(f_path)    

assert len(ais_cur1_path)!=0 
assert len(ais_cur2_path)!=0 
assert len(ais_lambda1_path)!=0 
assert len(ais_lambda2_path)!=0 
assert len(ais_lambda2_path)!=0 
assert len(ais_phi1_path)!=0 
assert len(ais_phi2_path)!=0 


def read_ais(year, month, day):
    ais_cur1 = {}
    ais_cur2 = {}
    ais_lambda1 = {}
    ais_lambda2 = {}
    ais_phi1 = {}
    ais_phi2 = {}

    for hour in range(24):
        dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)

        dtidx = date_to_dtidx(base_dt, dt)
        ais_cur1[dtidx] = []
        ais_cur2[dtidx] = []
        ais_lambda1[dtidx] = []
        ais_lambda2[dtidx] = []
        ais_phi1[dtidx] = []
        ais_phi2[dtidx] = []
        
        if dtidx in ais_cur1_path.keys():
            for i in range(len(ais_cur1_path[dtidx])):
                try:
                    ais_cur1[dtidx].append(pd.read_csv(ais_cur1_path[dtidx][i], encoding="cp932", header=None))
                    ais_cur1[dtidx][-1] = ais_cur1[dtidx][-1].values
                    #tf = ais_cur1[dtidx][-1]!=ais_cur1[dtidx][-1]
                    #ais_cur1[dtidx][-1][tf] = 0.0
                except:
                    print(f'Error load {ais_cur1_path[dtidx][i]}')
        
        if dtidx in ais_cur2_path.keys():
            for i in range(len(ais_cur2_path[dtidx])):
                try:
                    ais_cur2[dtidx].append(pd.read_csv(ais_cur2_path[dtidx][i], encoding="cp932", header=None))
                    ais_cur2[dtidx][-1] = ais_cur2[dtidx][-1].values
                    #tf = ais_cur2[dtidx][-1]!=ais_cur2[dtidx][-1]
                    #ais_cur2[dtidx][-1][tf] = 0.0 
                except:
                    print(f'Error load {ais_cur2_path[dtidx][i]}')
        
        
        if dtidx in ais_lambda1_path.keys():
            for i in range(len(ais_lambda1_path[dtidx])):
                try:
                    #print(f'load from {ais_lambda1_path[dtidx][i]}')
                    ais_lambda1[dtidx].append(pd.read_csv(ais_lambda1_path[dtidx][i], encoding="cp932", header=None))
                    ais_lambda1[dtidx][-1] = ais_lambda1[dtidx][-1].values  
                    #tf = ais_lambda1[dtidx][-1]!=ais_lambda1[dtidx][-1] 
                    #ais_lambda1[dtidx][-1][tf] = 1/1e10 
                except:
                    print(f'Error load {ais_lambda1_path[dtidx][i]}')

        if dtidx in ais_lambda2_path.keys():
            for i in range(len(ais_lambda2_path[dtidx])):
                try:
                    #print(f'load from {ais_lambda2_path[dtidx][i]}')
                    ais_lambda2[dtidx].append(pd.read_csv(ais_lambda2_path[dtidx][i], encoding="cp932", header=None))
                    ais_lambda2[dtidx][-1] = ais_lambda2[dtidx][-1].values  
                    #tf = ais_lambda2[dtidx][-1]!=ais_lambda2[dtidx][-1]
                    #ais_lambda2[dtidx][-1][tf] = 1/1e10 
                except:
                    print(f'Error load {ais_lambda2_path[dtidx][i]}')

        if dtidx in ais_phi1_path.keys():
            for i in range(len(ais_phi1_path[dtidx])):
                try:
                    ais_phi1[dtidx].append(pd.read_csv(ais_phi1_path[dtidx][i], encoding="cp932", header=None))
                    ais_phi1[dtidx][-1] = ais_phi1[dtidx][-1].values  
                    #tf = ais_phi1[dtidx][-1]!=ais_phi1[dtidx][-1]
                    #ais_phi1[dtidx][-1][tf] = 0.0 
                except:
                    print(f'Error load {ais_phi1_path[dtidx][i]}')

        if dtidx in ais_phi2_path.keys():
            for i in range(len(ais_phi2_path[dtidx])):
                try:
                    ais_phi2[dtidx].append(pd.read_csv(ais_phi2_path[dtidx][i], encoding="cp932", header=None))
                    ais_phi2[dtidx][-1] = ais_phi2[dtidx][-1].values  
                    #tf = ais_phi2[dtidx][-1]!=ais_phi2[dtidx][-1]
                    #ais_phi2[dtidx][-1][tf] = 0.0
                except:
                    print(f'Error load {ais_phi2_path[dtidx][i]}')

    return ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2

def read_jcope(year, month, day):
    jcope_n = {}
    jcope_e = {}

    for hour in range(24):
        dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        jcope_n[dtidx] = []
        jcope_e[dtidx] = []
        try:
            n = pd.read_csv(jcope_n_path[dtidx], encoding="cp932", header=None)
            n = n.values
            n[n!=n] = 0.0
            jcope_n[dtidx] = n
        except:
            print(f'Error load {jcope_n_path[dtidx]}')

        try:
            e = pd.read_csv(jcope_e_path[dtidx], encoding="cp932", header=None)
            e = e.values
            e[e!=e] = 0.0
            jcope_e[dtidx] = e
        except:
            print(f'Error load {jcope_e_path[dtidx]}')

    return jcope_n, jcope_e


# Load jcope and ais 
pm.printline('Loading ais and jcope now')
day = 1
ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2\
    = read_ais(dt_year, dt_month, day)

jcope_n, jcope_e = read_jcope(dt_year, dt_month, day)

pm.clear()
print(f'ais size: (ais x, ais y) = ({len(ais_cur1)}, {len(ais_cur2)})')
print(f'jcope size: (jcope n, jcope e) = ({len(jcope_n)}, {len(jcope_e)})')

dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
dtidx = date_to_dtidx(base_dt, dt)

TF = pkl.load(open('TF.pkl', 'rb'))

print(f'Target data num: {np.sum(TF)}')
n = _N = np.sum(TF) 
m = _M = 1
_NM = _N * _M
_2NM = 2 * _NM

# Define Kalaman func
def F_mat(jcope_n, jcope_e, dtidx):
    a = 0.5
    F = np.zeros((_2NM+1, _2NM+1))

    jcope0 = get_x(jcope_n, jcope_e, dtidx-1)
    jcope1 = get_x(jcope_n, jcope_e, dtidx)

    for i in range(_2NM):
        F[i][i] = a
        residual = jcope1[i] - a*jcope0[i]
        F[i][-1] = residual 
    F[_2NM][-1] = 1.0
    #print(f'F:{F}F')
    return F

# TODO
def H_mat(lambda1, lambda2, phi1, phi2):

    #min_value = 1/1e10
    H = np.zeros((_2NM+1, _2NM+1))
    for i in range(_N):
        H[i][i] = np.cos(phi1[i]+np.pi)
        H[i][i+_N] = np.sin(phi1[i]+np.pi)
        H[i+_N][i] = np.cos(phi1[i]+np.pi*3/2)
        H[i+_N][i+_N] = np.sin(phi1[i]+np.pi*3/2)
        # cos1 = np.cos(phi1[i])
        # sin1 = np.sin(phi1[i])
        # cos2 = np.cos(phi2[i])
        # sin2 = np.sin(phi2[i])
        # H[i][i] = cos1 if cos1==cos1 else min_value
        # H[i][i+_N] = sin1 if sin1==sin1 else min_value
        # H[i+_N][i] = cos2 if cos2==cos2 else min_value
        # H[i+_N][i+_N] = sin2 if sin2==sin2 else min_value 
    #print(f'H:{H}H')
    return H

        
    
def R_mat(sigma1, sigma2):

    # 大体悪い値で1/3位，最大を10.0に設定しておく
    #max_sigma = 10.0
    #sigma1[sigma1>max_sigma] = max_sigma
    #sigma2[sigma2>max_sigma] = max_sigma
    #sigma1[sigma1!=sigma1] = max_sigma
    #sigma2[sigma2!=sigma2] = max_sigma

    R = np.zeros((_2NM+1, _2NM+1))
    for i in range(_N):
        #R[i][i] = sigma1[i]
        #R[i+_N][i+_N] = sigma2[i]
        R[i][i] = 100
        R[i+_N][i+_N] = 100 
    #print(f'R:{R}R')
    return R

def get_z(ais_cur1, ais_cur2, dtidx, default=None):

    min_value = 0.0 #1/1e10
    ais_cur1_dt = kurosio_filter(ais_cur1[dtidx][0], nan_map)[TF] # 時刻0のais data
    ais_cur2_dt = kurosio_filter(ais_cur2[dtidx][0], nan_map)[TF] # 時刻0のais data

    ais_cur12_dt = np.concatenate([ais_cur1_dt, ais_cur2_dt])
    ais_cur12_dt = np.concatenate([ais_cur12_dt, [1]])
    ais_cur12_dt = ais_cur12_dt.reshape(_2NM+1, 1)

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
    jcope_ne_dt = jcope_ne_dt.reshape(_2NM+1, 1)
    jcope_ne_dt[np.isnan(jcope_ne_dt)] = 0.0
    return jcope_ne_dt


# Setting Kalman Param
pm.printline('Setting Kalman param now')
#kf = KalmanFilter(dim_x=_2NM+1, dim_z=_2NM+1)

## 初期の観測値と状態値
z = get_z(ais_cur1, ais_cur2, dtidx)
tf_z = np.isnan(z)
z[tf_z] = 0.0

x = jcope = get_x(jcope_n, jcope_e, dtidx)
assert np.sum(np.isnan(x))==0

## F関数
#kf.F = F_mat(jcope_n, jcope_e, dtidx+1)
F = F_mat(jcope_n, jcope_e, dtidx+1)

## H関数
lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)[TF]
lambda1[np.isnan(lambda1)] = 1.0
lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)[TF]
lambda2[np.isnan(lambda2)] = 1.0

phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)[TF]
phi1[np.isnan(phi1)] = 0.0
phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)[TF]
phi2[np.isnan(phi2)] = 0.0

#kf.H = H_mat(lambda1, lambda2, phi1, phi2)
H = H_mat(lambda1, lambda2, phi1, phi2)

## 分散共分散行列
#kf.Q = np.eye(_2NM+1)*0.1 # システム誤差(jcope)
#Q = np.eye(_2NM+1)*1000 # システム誤差(jcope)
Q = np.eye(_2NM+1)*0.00001 # システム誤差(jcope)

sigma1 = 1/(2*lambda1)
sigma2 = 1/(2*lambda2)
#kf.R = R_mat(sigma1, sigma2) # 観測誤差(ais)
R = R_mat(sigma1, sigma2) # 観測誤差(ais)

#kf.x = jcope_ne_dt0 # 初期位置
#kf.P = np.eye(_2NM+1) # 初期事後誤差共分散行列
P = np.eye(_2NM+1) # 初期事後誤差共分散行列
I = np.eye(_2NM+1) # 初期事後誤差共分散行列

## logの記録
pm.clear()
logger.record_tabular(f"dtidx", dtidx)
logger.record_tabular(f"Abailable_AIS", np.sum(z==z))
logger.record_tabular(f"Abailable_Jcope", np.sum(x==x))
logger.record_tabular(f"Abailable_Lambda", np.sum(lambda1==lambda1)+np.sum(lambda2==lambda2))
logger.record_tabular(f"Abailable_Sigma", np.sum(sigma1==sigma1)+np.sum(sigma2==sigma2))
logger.record_tabular(f"Abailable_Phi", np.sum(phi1==phi1)+np.sum(phi2==phi2))
logger.dump_tabular()


WHour = (1/2)**(1/8.5)

with open(f'{save_dir}/saverX{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(x, f)
with open(f'{save_dir}/saverZ{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(z, f)
with open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(jcope, f)

with open(f'{save_dir}/saverP{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(P, f)
with open(f'{save_dir}/saverR{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(R, f)
with open(f'{save_dir}/saverF{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(F, f)
with open(f'{save_dir}/saverH{year}{month:02}{day:02}00-N{_N}.pkl', 'wb') as f:
    pickle.dump(H, f)

# Filtering開始
print('Start KalmanFilter')
for day in range(1, n_day+1):
    dt = datetime.datetime(dt_year, dt_month, day)
    pm.printline(f'Filtering {dt}')
#for day in range(23, 23+1):
    # load ais and jcope here
    for t in tqdm.tqdm(range(1,23)):
        dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        if 0<t and t<24:
            F = F_mat(jcope_n, jcope_e, dtidx)

        # 予測ステップ
        pm.printline('Predicting now')
        #kf.predict()
        x = F @ x
        #x = x
        P = F @ P @ F.T + Q
        
        # フィルタリングステップ
        pm.printline('Filtering now')

        prev_z = z
        prev_lambda1 = lambda1
        prev_lambda2 = lambda2
        prev_phi1 = phi1
        prev_phi2 = phi2

        z = get_z(ais_cur1, ais_cur2, dtidx)
        tf = np.isnan(z)
        z[tf]= prev_z[tf]

        lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)[TF]
        tf = np.isnan(lambda1)
        lambda1[tf] = WHour * prev_lambda1[tf]

        lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)[TF]
        tf = np.isnan(lambda2)
        lambda2[tf] = WHour * prev_lambda2[tf]

        phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)[TF]
        tf = np.isnan(phi1)
        phi1[tf] = prev_phi1[tf] #TODO

        phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)[TF]
        tf = np.isnan(phi2)
        phi2[tf] = prev_phi2[tf] #TODO

        sigma1 = 1/(2*lambda1)
        sigma2 = 1/(2*lambda2)

        #kf.R = R_mat(sigma1, sigma2) # 観測誤差(ais)
        H = H_mat(lambda1, lambda2, phi1, phi2)
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
        #PHT = P @ H.T
        #S = H @ PHT + R
        S = I
        assert np.sum(np.isnan(S))==0

        inv = 'pinv'
        #SI = np.linalg.pinv(S.T)
        SI = I
        #K = PHT @ SI
        K = I

        y = z - H @ x
        #x = x + K @ y

        #I_KH = I - K @ H
        # TODO ちょっと知ってるのと違う，式的にRを写像してPに加えてる，Rの誤差も考慮するようにしてる?
        #P = I_KH @ P @ I_KH.T + K @ R @ K.T
        
        jcope = get_x(jcope_n, jcope_e, dtidx)
               
            

        # 関数の更新
        pm.printline('Updating function now')
        #kf.F = F_mat(jcope_n, jcope_e, dtidx+1)
        #kf.H = H_mat(lambda1, lambda2, phi1, phi2)

        pm.clear()
        logger.record_tabular(f"dtidx", dtidx)
        logger.record_tabular(f"SI", inv)
        logger.record_tabular(f"Abailable_AIS", np.sum(z==z))
        logger.record_tabular(f"Abailable_Jcope", np.sum(x==x))
        logger.record_tabular(f"Abailable_Lambda", np.sum(lambda1==lambda1)+np.sum(lambda2==lambda2))
        logger.record_tabular(f"Abailable_Sigma", np.sum(sigma1==sigma1)+np.sum(sigma2==sigma2))
        logger.record_tabular(f"Abailable_Phi", np.sum(phi1==phi1)+np.sum(phi2==phi2))
        logger.dump_tabular()

        # 保存
        with open(f'{save_dir}/saverX{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(x, f) #KALMAN
        with open(f'{save_dir}/saverZ{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(z, f) #AIS
        with open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(jcope, f) #JCOPE
        with open(f'{save_dir}/saverP{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(P, f)
        with open(f'{save_dir}/saverR{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(R, f)
        with open(f'{save_dir}/saverF{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(F, f)
        with open(f'{save_dir}/saverH{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(H, f)
        with open(f'{save_dir}/saverK{year}{month:02}{day:02}{t:02}-N{_N}.pkl', 'wb') as f:
            pickle.dump(K, f)

        

    ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2\
        = read_ais(dt_year, dt_month, day)
    
    jcope_n, jcope_e = read_jcope(dt_year, dt_month, day)

print('Finished KalmanFilter')

    
