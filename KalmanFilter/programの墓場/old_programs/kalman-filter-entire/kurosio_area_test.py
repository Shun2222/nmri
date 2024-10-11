
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
from entire_utils import *
from entire_kf_params import *
import logger
import printManager as pm

from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from Ais4ToCur import ais4
import Ais4ToCur as atc

# パラメータの設定
year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'

jl = JCOPELoader(year, month)
jl.load_path(path_jcope)

day = 1
hour = 1
dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
dtidx = date_to_dtidx(base_dt, dt)

jcope_n, jcope_e = jl.load_jcope_day(day)
print(jcope_e[dtidx].shape)
jcope_e = kurosio_filter_pooled(jcope_e[dtidx], nan_map_pooled) # 時刻0のjcope data
print(jcope_e.shape)
jn_map = kurosio_vec_to_map_pooled(jcope_e, nan_map_pooled) * nan_map_pooled
print(jn_map.shape)
# jn_map = jcope_e[dtidx]
df_jn_map = pd.DataFrame(jn_map)

path = osp.join(r'./data', f'kurosio-area-test_jcope_e{year}{month:02}{day:02}{hour:02}.csv')
df_jn_map.to_csv(path, index=False, header=False)