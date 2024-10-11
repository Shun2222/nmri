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

path_ship = path = r"E:\shunsukeE\data\shiplog/"

files = os.listdir(path)
target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]

for target_ship in target_ships:
    #if target_ship in done_ship:
    #    print(f'skip {target_ship}')
    #    continue

    for day in range(1, n_day+1):
        #print(f'{target_ship} {dt_month:02}{day:02}')
        dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)

        f_name = fr'cur_hours{dt_year}{dt_month:02}{day:02}.csv'
        f_path = osp.join(path_ship, target_ship, '2015', f_name)
        path_log = osp.join(path_ship, target_ship, '2015')
        try:
            shipLog = pd.read_csv(f_path, encoding="cp932")
            shipLog['ShipName'] = target_ship

            if len(shipLog)==0:
                print(f'{target_ship} {dt_month:02}{day:02}: Not exist data')
                continue

        except:
            print(f'{target_ship} {dt_month:02}{day:02}: Error load {f_path}')
            continue
