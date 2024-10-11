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
import seaborn as sns
import japanize_matplotlib

from ais_loader import AISLoader
from kalmanLog_loader import KalmanLogLoader
from jcope_loader import JCOPELoader
from utils import *
from kf_params import *



year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) - 1
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []
is_light = True

path_ship = path = r"E:\shunsukeE\data\shiplog/"

files = os.listdir(path)
target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]

datamap = np.zeros(nan_map_pooled.shape) * nan_map_pooled
datamap_ship = {}
for target_ship in target_ships:
    datamap_ship[target_ship] = np.zeros(nan_map_pooled.shape) * nan_map_pooled

for day in range(1, n_day+1):
#for day in range(23, 23+1):
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    print(dt)
    
    for target_ship in target_ships:
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

        for i in range(len(shipLog)):
            time = shipLog['DtIdx'].values[i]
            grid0 = shipLog['Grid0'].values[i]
            grid1 = shipLog['Grid1'].values[i]
            curN = shipLog['CurN'].values[i]
            curE = shipLog['CurE'].values[i]
            dtidx = time

            grid0 = int(grid0/2)
            grid1 = int(grid1/2)
            # if not kurosio_map_tf[grid0][grid1]:
            #     print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
            #     continue
            #idx = int(kurosio_index[grid0][grid1])
            # if idx==-1:
            #     print(f'Out of idx ( grid = ({grid0},{grid1}) ).')
            #     continue

            if dtidx<0:
                print(f'Out of dtidx ( dtidx = {dtidx}) ).')
                continue

            
            datamap[grid0][grid1] += 1
            datamap_ship[target_ship][grid0][grid1] += 1
        
for target_ship in target_ships:
    path = fr'./Images/{target_ship}-datamap.png'
    sns.heatmap(datamap_ship[target_ship])
    count = np.nansum(datamap_ship[target_ship])
    plt.title(f'{target_ship} n_data={count}')
    plt.savefig(path)
    print(f'saved {path}')
    plt.close()
path = fr'./Images/datamap.png'
sns.heatmap(datamap)
plt.savefig(path)
plt.close()
print(f'saved {path}')
