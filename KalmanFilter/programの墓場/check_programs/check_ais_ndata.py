
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
import seaborn as sns
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from Ais4ToCurForKalmanTime import ais4
import Ais4ToCurForKalmanTime as atc
from utils import * 

lat00 = int(map_pooled_size[0]-kurosio_latidx_range1[0]/2)
lat11 = int(map_pooled_size[0]-kurosio_latidx_range2[1]/2)
lon0 = int(map_pooled_size[1]-kurosio_lonidx_range[0]/2)
lon1 = int(map_pooled_size[1]-kurosio_lonidx_range[1]/2)
print(f'lat00 {lat00}')
print(f'lat11 {lat11}')
print(f'lon0 {lon0}')
print(f'lon1 {lon1}')

year = 2015
month = 9
n_day = nday_month(month) - 1
n_hour = 24 #24
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True

print(f'Loading data')

kl = KalmanLogLoader(2015, 9, 20)
log_path = r"E:\shunsukeE\result\kalman-west-kurosio-Q001Q01"
path_ais = r'E:\shunsukeE\data\ais\1509-ais4s-pkls'
kalman_keys = ['X', 'JCOPE', 'Z']
kl.set_path(log_path)

ais_keys = ['n']
al = atc.AISLoader(year, month, osp.join(path_ais, 'ais_files'))
al.set_keys(ais_keys)

def make_path(p1, p2, extension=None):
    path = osp.join(p1, p2)
    if extension:
        path = path+extension
    return path

for s in range(0, 1):
    n_datas = []
    e_datas = []
    for day in tqdm.tqdm(range(1, n_day+1)):
        print(f's:{s}, day:{day}')
        # TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']
        if is_light:
            TF = np.array([True] * 3789)
        else:
            TF = np.array([True] * 9808)

        start_hour = 0 if day == 1 else 1
        for hour in range(start_hour, n_hour):
            date_str = f'{month}/{day} {hour}:00'
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue

            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue
            
            data  = al.load_cur(dtidx) 
            an_map = [data[key] for key in ais_keys][0]

            lat00 = int(map_pooled_size[0]-kurosio_latidx_range1[0]/2)
            lat11 = int(map_pooled_size[0]-kurosio_latidx_range2[1]/2)
            lon0 = int(map_pooled_size[1]-kurosio_lonidx_range[0]/2)
            lon1 = int(map_pooled_size[1]-kurosio_lonidx_range[1]/2)
            dif = 3 

            an_map = an_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            an_map[an_map==0] = 1 
            n_data = np.nansum(an_map/an_map)
            print(fr'{date_str}, データ数: {n_data}')
            n_datas.append(n_data)

x = np.arange(len(n_datas))
plt.plot(x, n_datas)        
plt.title('Num data of AIS')
plt.xlabel(f'hours')
plt.ylabel(f'num data')
path = f'Num_data_ais.png'
path = osp.join(log_path, path)
plt.savefig(path)
print(n_datas)
