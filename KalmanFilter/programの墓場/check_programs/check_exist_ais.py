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
from ais_loader import AISLoader
import matplotlib.pyplot as plt

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
n_day = 2#nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)

path_ais = fr'E:\shunsukeE\data\ais'


# AISのファイルパスの読み取り
#keys = ['cur1', 'cur2', 'lambda1', 'lambda2', 'phi1', 'phi2']
keys = ['cur1', 'cur2']
al = AISLoader(2015, 9)
al.load_path(keys=keys)


dt = datetime.datetime(dt_year, dt_month, 1, 0, 0, 0)
dtidx = date_to_dtidx(base_dt, dt)
data = al.load_ais_day(1)
#ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2 = [data[key] for key in keys]
ais_cur1, ais_cur2 = [data[key] for key in keys]
dt = datetime.datetime(dt_year, dt_month, 1, 0, 0, 0)
cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map) # 時刻0のais data
cur2 = kurosio_filter(ais_cur2[dtidx][0], nan_map) # 時刻0のais data
#lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)
#lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)
#phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)
#phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)

tf1 = cur1==cur1
tf2 = cur2==cur2
cur_tf = tf1 & tf2
cur_ndata = np.zeros(cur_tf.shape)
cur_ndata[cur_tf] = 1

#tf1 = lambda1==lambda1
#tf2 = lambda1==lambda1
#lambda_tf = tf1 & tf2
#lambda_ndata = np.zeros(lambda_tf.shape)
#lambda_ndata[lambda_tf] = 1
#
#tf1 = phi1==phi1
#tf2 = phi1==phi1
#phi_tf = tf1 & tf2
#phi_ndata = np.zeros(phi_tf.shape)
#phi_ndata[phi_tf] = 1


for day in range(1, n_day):
    data = al.load_ais_day(day)
    #ais_cur1, ais_cur2, ais_lambda1, ais_lambda2, ais_phi1, ais_phi2 = [data[key] for key in keys]
    ais_cur1, ais_cur2 = [data[key] for key in keys]
    for t in tqdm.tqdm(range(1,24)):
        dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        if not dtidx in ais_cur1.keys():
            continue
        if ais_cur1[dtidx]==0:
            continue

        cur1 = kurosio_filter(ais_cur1[dtidx][0], nan_map) # 時刻0のais data
        cur2 = kurosio_filter(ais_cur2[dtidx][0], nan_map) # 時刻0のais data
#        lambda1 = kurosio_filter(ais_lambda1[dtidx][0], nan_map)
#        lambda2 = kurosio_filter(ais_lambda2[dtidx][0], nan_map)
#        phi1 = kurosio_filter(ais_phi1[dtidx][0], nan_map)
#        phi2 = kurosio_filter(ais_phi2[dtidx][0], nan_map)

        tf1 = cur1==cur1
        tf2 = cur2==cur2
        tf = tf1 & tf2
        cur_tf = np.vstack((cur_tf, tf))
        cur_one = np.zeros(tf.shape)
        cur_one[tf] = 1
        cur_ndata += cur_one

#        tf1 = lambda1==lambda1
#        tf2 = lambda2==lambda2
#        tf = tf1 & tf2
#        lambda_tf = np.vstack((lambda_tf, tf))
#        lambda_one = np.zeros(tf.shape)
#        lambda_one[tf] = 1
#        lambda_ndata += lambda_one
#
#        tf1 = phi1==phi1
#        tf2 = phi2==phi2
#        tf = tf1 & tf2
#        phi_tf = np.vstack((phi_tf, tf))
#        phi_one = np.zeros(tf.shape)
#        phi_one[tf] = 1
#        phi_ndata += phi_one


print(f'cur: {np.sum(cur_tf, axis=0)}')
#print(f'lambda: {np.sum(lambda_tf, axis=0)}')
#print(f'phi: {np.sum(phi_tf, axis=0)}')

pkl.dump(cur_tf, open('data/cur_tf.pkl', 'wb'))
#pkl.dump(lambda_tf, open('lambda_tf.pkl', 'wb'))
#pkl.dump(phi_tf, open('phi_tf.pkl', 'wb'))

pkl.dump(cur_ndata, open('data/cur_ndata.pkl', 'wb'))
#pkl.dump(lambda_ndata, open('lambda_ndata.pkl', 'wb'))
#pkl.dump(phi_ndata, open('phi_ndata.pkl', 'wb'))

is_exist = cur_tf
y = np.sum(is_exist, axis=1)
x = np.arange(len(y)) 
plt.plot(x, y)
plt.title('num data 9/1-9/30')
plt.xlabel('Time')
plt.ylabel('ndata')
plt.savefig('Images/ndata-time.png')
plt.close()

y = np.sum(is_exist, axis=0)
x = np.arange(len(y)) 
plt.plot(x, y)
plt.title('num data each kuroshio index 9/1-9/30')
plt.xlabel('Index')
plt.ylabel('ndata')
plt.savefig('Images/ndata-eachIdx.png')
plt.close()
