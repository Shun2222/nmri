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


#パラメータの設定
year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = 2#nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'
path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"
save_dir = r"E:\shunsukeE\result\kalman-test-jcope_ais0001"
logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])

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

# Load jcope and ais 
pm.printline('Loading ais and jcope now')

dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2\
    = read_ais(dt_year, dt_month, day)
cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map) # 時刻0のais data
cur2 = kurosio_filter(ais_cur2[dtidx][0], nan_map) # 時刻0のais data
lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)
lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)
phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)
phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)

tf1 = cur1==cur1
tf2 = cur2==cur2
cur_tf = tf1 & tf2

tf1 = lambda1==lambda1
tf2 = lambda1==lambda1
lambda_tf = tf1 & tf2

tf1 = phi1==phi1
tf2 = phi1==phi1
phi_tf = tf1 & tf2

#for day in range(1, n_day+1):
for day in range(1, 2):
    dt = datetime.datetime(dt_year, dt_month, day)
    pm.printline(f'Check existing data {dt}')
    ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2\
        = read_ais(dt_year, dt_month, day)
    for t in tqdm.tqdm(range(1,23)):
        dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        print(dtidx)

        cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map) # 時刻0のais data
        cur2 = kurosio_filter(ais_cur2[dtidx][0], nan_map) # 時刻0のais data

        lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)
        lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)

        phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)
        phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)

        tf1 = cur1==cur1
        tf2 = cur2==cur2
        cur_tf = tf1 & tf2

        tf1 = lambda1==lambda1
        tf2 = lambda1==lambda1
        lambda_tf = tf1 & tf2

        tf1 = phi1==phi1
        tf2 = phi1==phi1
        phi_tf = tf1 & tf2

sp_cur_tf = split_into_ranges(np.where(cur_tf)[0])
for i, sp in enumerate(sp_cur_tf):
    print(f'{i}: {len(sp)}')

idx = input('select idx:')
if idx!='':
    idx = int(idx)
    tf = [False for _ in range(len(cur_tf))]
    for spi in sp_cur_tf[idx]:
        tf[spi] = True

pkl.dump(tf, open('result/TF.pkl', 'wb'))
pkl.dump(cur_tf, open('result/cur_tf.pkl', 'wb'))
pkl.dump(sp_cur_tf, open('result/sp_cur_tf.pkl', 'wb'))
pkl.dump(cur_tf, open('result/cur_tf.pkl', 'wb'))
pkl.dump(lambda_tf, open('result/lambda_tf.pkl', 'wb'))
pkl.dump(phi_tf, open('result/phi_tf.pkl', 'wb'))
