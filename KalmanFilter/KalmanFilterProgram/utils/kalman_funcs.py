import datetime
import numpy as np
import pandas as pd
from numpy.random import randn
import pickle as pkl
import math
import os 
import os.path as osp
import re
import scipy as sp
import scipy.sparse as sps

from utils import *
from utils.kalman_parameters import *
from utils.utils_needed_params import *

# filteringに使う関数の宣言
def F_mat(jcope0, jcope1, _2N0):
    a = 0.5
    F = sps.lil_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32))

    for i in range(_2N0):
        F[i, i] = a
        residual = jcope1[i, 0] - a*jcope0[i, 0]
        F[i, -1] = np.float32(residual)
    F[_2N0, -1] = np.float32(1.0)
    # print(f'F:{F}F')
    return F.tocsr()

def B_mat(_N0):
    _2N0 = 2*_N0
    B = sps.lil_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32))
    range_lat = 5 # TODO
    range_lon = 10 # TODO
    lat_halflife = 0.037
    lon_halflife = 0.064
    deg_per_mesh = 1/(36/pool_size)
    theta = -np.arctan(1/3)
    for i in range(_N0):
        grid0, grid1 = kurosio_grid_pooled[i]
        # lat', lon'方向に対して重みを計算していく
        for lat in np.arange(0, range_lat, 0.3):
            for lon in np.arange(0, range_lon, 0.3):
                # 以下、±４方向でそれぞれ重みを計算し、行列Bに反映

                def w_elem(dlat, dlon):
                    w = ((1/2) ** (dlat/lat_halflife)) * ((1/2) ** (dlon/lon_halflife))
                    return w

                # 方向１ (++)
                # 黒潮方向dy2とdy2+90°方向dx2を計算
                dy2 = int(lat*np.cos(theta) + lon*np.sin(theta))
                dx2 = int(lon*np.cos(theta) - lat*np.sin(theta))
                # マス目に対してのlat',lon'方向の大きさを度数単位で計算
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = w_elem(dydeg, dxdeg)
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w

                # 方向2 (+-)
                dy2 = int(lat*np.cos(theta) - lon*np.sin(theta))
                dx2 = int(-lon*np.cos(theta) - lat*np.sin(theta))
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = w_elem(dydeg, dxdeg)
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w

                # 方向3 (-+)
                dy2 = int(-lat*np.cos(theta) + lon*np.sin(theta))
                dx2 = int(lon*np.cos(theta) + lat*np.sin(theta))
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = w_elem(dydeg, dxdeg)
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w

                # 方向4 (--)
                dy2 = int(-lat*np.cos(theta) - lon*np.sin(theta))
                dx2 = int(-lon*np.cos(theta) + lat*np.sin(theta))
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = w_elem(dydeg, dxdeg)
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w
    for i in range(_2N0):
        B[i, i] = 1.0
    return B.tocsr()

def H_mat(phi, idxs, _N, _N0):
    _2N0 = 2*_N0
    # min_value = 1/1e10
    H = sps.lil_matrix(np.zeros((_N+1, _2N0+1), dtype=np.float32))
    for i in range(_N):
        idx = idxs[i]
        H[i, idx] = np.cos(phi[i])
        H[i, idx+_N0] = np.sin(phi[i])
    H[-1, -1] = 1.0
    length = H.shape[0] # これを入れるとerrorなくなる (配列の長さを計算して内部で保持してる？)
    return H
    
def R_mat(sigma2, _N):

    R = sps.lil_matrix(np.zeros((_N+1, _N+1), dtype=np.float32))
    for i in range(_N):
        R[i, i] = sigma2[i]
    return R.tocsr()

def get_z(ais_cur2, dtidx, _N0, default=None):

    if len(ais_cur2)==0: 
        print(f'Not exist data({dtidx}) in ais cur 2')
        nan_data = [np.nan for _ in range(_N0)] + [1]
        return np.array(nan_data) 

    ais_cur2_dt = kurosio_filter_pooled(ais_cur2, nan_map_pooled, is_pooled=True) # 時刻0のais data
    ais_cur12_dt = np.concatenate([ais_cur2_dt, [1]])
    ais_cur12_dt = ais_cur12_dt.reshape(_N0+1, 1)

    tf = np.isnan(ais_cur12_dt)
    if default==None:
        ais_cur12_dt[tf] = np.nan 
    else:
        ais_cur12_dt[tf] = default[tf]
    return sps.csr_matrix(ais_cur12_dt)

def get_x(jcope_n, jcope_e, dtidx, _2N0, default=np.nan):

    jcope_n_dt = kurosio_filter_pooled(jcope_n[dtidx], nan_map_pooled) # 時刻0のjcope data
    jcope_n_dt = jcope_n_dt # 時刻0のjcope data

    jcope_e_dt = kurosio_filter_pooled(jcope_e[dtidx], nan_map_pooled) # 時刻0のjcope data

    jcope_ne_dt = np.concatenate([jcope_n_dt, jcope_e_dt])
    jcope_ne_dt = np.concatenate([jcope_ne_dt, [1]])
    jcope_ne_dt = jcope_ne_dt.reshape(_2N0+1, 1)
    jcope_ne_dt[np.isnan(jcope_ne_dt)] = 0.0
    return sps.csr_matrix(jcope_ne_dt)