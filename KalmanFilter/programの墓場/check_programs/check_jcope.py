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
save_dir = r"E:\shunsukeE\result\kalman09"
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
assert len(jcope_n_path)!=0
assert len(jcope_e_path)!=0

