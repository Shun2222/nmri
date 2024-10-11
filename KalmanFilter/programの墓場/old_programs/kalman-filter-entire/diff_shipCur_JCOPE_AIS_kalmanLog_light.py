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

from kalmanLog_loader import KalmanLogLoader
from jcope_loader import JCOPELoader
from experiment_utils import *
from experiment_kf_params import *



year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) - 1
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []
is_light = True
s = 'v2' #v2 or 0

path_jcope = fr'E:\shunsukeE\data\eas2'
path_ship = path = r"E:\shunsukeE\data\shiplog/"
path_kalman = r"E:\shunsukeE\result\kalman-pooled3-Q0.1"

#path_ais = fr'E:\shunsukeE\data\ais'
#path_ais = r'E:/shunsukeE/data/ais/ais_files'
path_ais = r'E:\shunsukeE\data\ais\1509-ais4s-pkls-pooled3'
num_slices = 1 


kalman_keys = ['X']
kl = KalmanLogLoader(year, month, 20)
kl.set_path(path_kalman)
data = kl.load_kalmanLog_day(1, s, keys=['X'])
if is_light:
    TFs = None
    n_data = -1
else:
    TFs = [np.array([True]*9808)]
    n_data = 9808
#TFs = [kl.load_kalmanLog_day(1, s, keys=['TF'])['TF'] for s in range(14)]

files = os.listdir(path)
target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]

df_aj = []
df_as = []
df_ak = []
df_sj = []
df_kj = []
df_ks = []
dtidxs = []
grids = []
diff_ship = {}
for target_ship in target_ships:
    diff_ship[target_ship] = [[], [], 0]

for day in range(1, n_day+1):
#for day in range(23, 23+1):
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    print(dt)
    
    for target_ship in target_ships:
        available_count = 0
        print(f'{target_ship} {dt_month:02}{day:02}')
        if target_ship in done_ship:
            print(f'skip {target_ship}')
            continue

        f_name = fr'cur_hours{dt_year}{dt_month:02}{day:02}.csv'
        #f_name = fr'cur_minutes{dt_year}{dt_month:02}{day:02}.csv'
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

        print(f'shiplog {len(shipLog)}')


        def get_kfidx(TF, idx):
            return np.sum(TF[:idx])-1

        data = kl.load_kalmanLog_day(day, s, keys=['X', 'JCOPE'])
        diffs = []
        for i in range(len(shipLog)):
            time = shipLog['DtIdx'].values[i]
            grid0 = shipLog['Grid0'].values[i]
            grid1 = shipLog['Grid1'].values[i]
            curN = shipLog['CurN'].values[i]
            curE = shipLog['CurE'].values[i]
            dtidxs.append(time)
            grids.append([grid0, grid1])
            dtidx = time

            if is_light:
                grid0 = int(grid0/pool_size)
                grid1 = int(grid1/pool_size)
                # if not kurosio_pooled(grid0, grid1):
                if not kurosio_map_tf_pooled[grid0][grid1]:
                    print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                    continue
                idx = int(kurosio_index_pooled[grid0][grid1])
            else:
                # if not kurosio(grid0, grid1):
                if not kurosio_map_tf[grid0][grid1]:
                    print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                    continue
                idx = int(kurosio_index[grid0][grid1])

            if dtidx<0:
                print(f'Out of dtidx ( dtidx = {dtidx}) ).')
                continue

            if idx==-1:
                print(f'Out of idx ( grid = ({grid0},{grid1}) ).')
                continue


            data = kl.load_kalmanLog_day(day, s, keys=['X', 'JCOPE'])
            if n_data == -1:
                n_data = int((data['X'][1].shape[0] - 1)/2)
                TFs = [np.array([True]*n_data)]

            #if not dtidx in data['X'].keys() and day!=1:
            if not dtidx in data['X'].keys():
                if day==1:
                    print(f'There is no kalman data (dtidx = {dtidx}).')
                    continue
                data = kl.load_kalmanLog_day(day-1, s, keys=['X', 'JCOPE'])
                if not dtidx in data['X'].keys():
                    print(f'There is no kalman data (dtidx = {dtidx}).')
                    continue

            kfidx = get_kfidx(TFs[0], idx)
            kalman_cur1 = data['X'][dtidx][kfidx]
            kalman_cur2 = data['X'][dtidx][kfidx+n_data]

            jcope_cur1 = data['JCOPE'][dtidx][kfidx]
            jcope_cur2 = data['JCOPE'][dtidx][kfidx+n_data]

            hour = dtidx_to_date(base_dt, int(dtidx)).hour
            path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}X.csv"
            dataN = pd.read_csv(path, encoding="cp932", header=None)
            ais_cur1 = dataN.values[grid0][grid1]
            
            path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}Y.csv"
            dataE = pd.read_csv(path, encoding="cp932", header=None)
            ais_cur2 = dataE.values[grid0][grid1]
            
            if ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                df_aj.append([ais_cur1-jcope_cur1, ais_cur2-jcope_cur2])
                df_as.append([ais_cur1-curN, ais_cur2-curE])
                df_ak.append([ais_cur1-kalman_cur1, ais_cur2-kalman_cur2])
            df_sj.append([jcope_cur1-curN, jcope_cur2-curE])    
            df_kj.append([jcope_cur1-kalman_cur1, jcope_cur2-kalman_cur2])    
            df_ks.append([kalman_cur1-curN, kalman_cur2-curE]) 
            diffs.append([kalman_cur1-curN, kalman_cur2-curE])
            available_count += 1 

        if available_count>0:
            diffs = np.array(diffs)
            diffs = diffs.reshape(diffs.shape[0], diffs.shape[1])
            diff_N = diffs.T[0]
            diff_E = diffs.T[1]
            diff_ship[target_ship][0] += diff_N.tolist()
            diff_ship[target_ship][1] += diff_E.tolist()
            diff_ship[target_ship][2] += available_count
            

print(f'aj: {np.mean(np.abs(df_aj))}')
print(f'ak: {np.mean(np.abs(df_ak))}')
print(f'as: {np.mean(np.abs(df_as))}')
print(f'kj: {np.mean(np.abs(df_kj))}')
print(f'sj: {np.mean(np.abs(df_sj))}')
print(f'ks: {np.mean(np.abs(df_ks))}')
for target_ship in target_ships:
    a = np.mean(diff_ship[target_ship][0])
    b = np.mean(diff_ship[target_ship][1])
    c = diff_ship[target_ship][2]
    print(f'{target_ship}: {round(np.mean(a), 2)}, {round(np.mean(b), 2)}, {c}')
