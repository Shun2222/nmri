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
n_day = 2 #nday_month(dt_month) 
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
ais_N_path = {} #固有ベクトルの方向2
ais_E_path = {} #固有ベクトルの方向2

patterns = [f'(\w+){month:02}..[0-9][0-9]Cur1.csv', 
            f'(\w+){month:02}..[0-9][0-9]Cur2.csv', 
            f'(\w+){month:02}..[0-9][0-9]Lambda1.csv',
            f'(\w+){month:02}..[0-9][0-9]Lambda2.csv',
            f'(\w+){month:02}..[0-9][0-9]Phi1.csv',
            f'(\w+){month:02}..[0-9][0-9]Phi2.csv',
            f'(\w+){month:02}..[0-9][0-9]X.csv',
            f'(\w+){month:02}..[0-9][0-9]Y.csv']

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

        if re.match(patterns[6], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_N_path.keys():
                ais_N_path[dtidx] = []
                ais_N_path[dtidx].append(f_path)
            else:
                ais_N_path[dtidx].append(f_path)    

        if re.match(patterns[7], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_E_path.keys():
                ais_E_path[dtidx] = []
                ais_E_path[dtidx].append(f_path)
            else:
                ais_E_path[dtidx].append(f_path)    

def read_ais(year, month, day):
    ais_cur1 = {}
    ais_cur2 = {}
    ais_lambda1 = {}
    ais_lambda2 = {}
    ais_phi1 = {}
    ais_phi2 = {}
    ais_N = {}
    ais_E = {}

    for hour in range(24):
        dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)

        dtidx = date_to_dtidx(base_dt, dt)
        ais_cur1[dtidx] = []
        ais_cur2[dtidx] = []
        ais_lambda1[dtidx] = []
        ais_lambda2[dtidx] = []
        ais_phi1[dtidx] = []
        ais_phi2[dtidx] = []
        ais_N[dtidx] = []
        ais_E[dtidx] = []
        
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
        else:
            print('No N data')
        if dtidx in ais_phi2_path.keys():
            for i in range(len(ais_phi2_path[dtidx])):
                try:
                    ais_phi2[dtidx].append(pd.read_csv(ais_phi2_path[dtidx][i], encoding="cp932", header=None))
                    ais_phi2[dtidx][-1] = ais_phi2[dtidx][-1].values  
                    #tf = ais_phi2[dtidx][-1]!=ais_phi2[dtidx][-1]
                    #ais_phi2[dtidx][-1][tf] = 0.0
                except:
                    print(f'Error load {ais_phi2_path[dtidx][i]}')

        if dtidx in ais_N_path.keys():
            for i in range(len(ais_N_path[dtidx])):
                try:
                    ais_N[dtidx].append(pd.read_csv(ais_N_path[dtidx][i], encoding="cp932", header=None))
                    ais_N[dtidx][-1] = ais_N[dtidx][-1].values  
                    #tf = ais_N[dtidx][-1]!=ais_N[dtidx][-1]
                    #ais_N[dtidx][-1][tf] = 0.0
                except:
                    print(f'Error load {ais_N_path[dtidx][i]}')


        if dtidx in ais_E_path.keys():
            for i in range(len(ais_E_path[dtidx])):
                try:
                    ais_E[dtidx].append(pd.read_csv(ais_E_path[dtidx][i], encoding="cp932", header=None))
                    ais_E[dtidx][-1] = ais_E[dtidx][-1].values  
                    #tf = ais_E[dtidx][-1]!=ais_E[dtidx][-1]
                    #ais_E[dtidx][-1][tf] = 0.0
                except:
                    print(f'Error load {ais_E_path[dtidx][i]}')
    return ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2, ais_N, ais_E

assert len(ais_N_path)!=0
day = 1
ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2, ais_N, ais_E\
    = read_ais(dt_year, dt_month, day)
# Load jcope and ais 
pm.printline('Loading ais and jcope now')

print(ais_cur1.keys())
dt = datetime.datetime(dt_year, dt_month, 1, 1, 0, 0)
dtidx = date_to_dtidx(base_dt, dt)
cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map) # 時刻0のais data
cur2 = kurosio_filter(ais_cur2[dtidx][0], nan_map) # 時刻0のais data
lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)
lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)
phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)
phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)
N = kurosio_filter(ais_N[dtidx][0], nan_map)
E = kurosio_filter(ais_E[dtidx][0], nan_map)

TF = pkl.load(open('TF.pkl', 'rb'))
i = 0 
print(f'cur1: {cur1[TF][i]}')
print(f'cur2: {cur2[TF][i]}')

print(f'cur1: {(N[TF][i]*np.cos(phi1[TF][i]) + E[TF][i]*np.sin(phi1[TF][i]))/lambda1[TF][i]}')
print(f'cur2: {(N[TF][i]*np.cos(phi2[TF][i]) + E[TF][i]*np.sin(phi2[TF][i]))/lambda2[TF][i]}')


print(f'cur1: {(N[TF][i]*np.cos(phi1[TF][i]) + E[TF][i]*np.sin(phi1[TF][i]))}')
print(f'cur2: {(N[TF][i]*np.cos(phi2[TF][i]) + E[TF][i]*np.sin(phi2[TF][i]))}')
