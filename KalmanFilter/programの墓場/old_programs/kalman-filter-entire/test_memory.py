###
# Kalman Filter 
# フィルタ結果と船の推定偏流との誤差を計算し、船の信頼度を更新する
###
print('\rimport files now', end='')
import sys
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
import scipy as sp
import scipy.sparse as sps
from memory_profiler import profile

from tqdm import tqdm
from entire_utils import *
from entire_kf_params import *
import logger
import printManager as pm

from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from Ais4ToCurForKalmanTime import ais4
import Ais4ToCurForKalmanTime as atc


@profile
def test():
    Q_values = [0.1]
    for Q_value in Q_values:
        pm.clear()
        # パラメータの設定
        year = dt_year = 2015
        month = dt_month = 9
        dt_day = 1
        n_day = nday_month(dt_month) 
        base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)

        N_lambda = 2000 # 使用するデータの最大数（lambdaを章順に並べ替えたときの上位N_lambda個を使用）
        Min_lambda = 10 # 使用するlambdaの下限

        path_jcope = fr'E:\shunsukeE\data\eas2' # jcopeデータの保存先
        path_ais = fr'E:\shunsukeE\data\ais' # aisデータの保存先
        save_dir = fr"E:\shunsukeE\result\kalman-entire-pooled3-Q{Q_value}" # データの保存先

        logger.record_tabular(f"Q value", Q_value)
        logger.record_tabular(f"N lambda", N_lambda)
        logger.record_tabular(f"Min lambda", Min_lambda)
        logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])

        # データの読み込みクラスの設定
        pm.printline('Setting jcope loader')
        jl = JCOPELoader(year, month)
        jl.load_path(path_jcope)

        pm.printline('Setting ais loader')
        ais_keys = ['cur1', 'lambda1', 'phi1', 'cur2', 'lambda2', 'phi2']
        ais_outfiles = save_dir + r"\ais_files"
        ais_path = r'E:/shunsukeE/data/ais/1509-ais4s-pkls-pooled3-entire'
        al = atc.AISLoader(year, month, ais_outfiles, pkl_path=ais_path)
        al.set_keys(ais_keys)

        pm.printline('Setting Kalman patameter')
        # isTarget = pkl.load(open(osp.join('./data/cur_ndata.pkl'), 'rb'))
        # _Targets = np.where(isTarget>0)[0]
        # enough_data_area = isTarget = pkl.load(open(osp.join('./data/ndataOver600.pkl'), 'rb'))

        # filteringの前準備
        pm.clear()
        for r in range(1):
            pm.printline(f'Loading ais and jcope now (n_slice={r})')
            atc.svm.clear() # 船毎の分散のリセット

            day = 1 
            dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)

            # AISの読み込み
            data  = al.load_cur(dtidx)
            ais_cur1, ais_lambda1, ais_phi1, ais_cur2, ais_lambda2, ais_phi2\
                = [data[key] for key in ais_keys]
            cur2 = kurosio_filter_pooled(ais_cur2, nan_map_pooled, is_pooled=True)# 時刻0のais data
            print(f'ais shape: {ais_cur2.shape}')

            # データ数の設定
            n = _N = _N0 = len(cur2)
            m = _M = 1
            _NM = _N * _M
            _2NM = _2N0 = 2 * _NM

            TF = np.array([True for _ in range(len(cur2))])


            #pkl.dump(TF, open(f'{save_dir}/Targets{year}{month:02}{day:02}-{r}.pkl', 'wb'))
            print(f'Target data num: {_2NM}')

            # filteringに使う関数の宣言
            def F_mat(jcope0, jcope1):
                a = 0.5
                F = sps.lil_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32))

                for i in range(_2N0):
                    F[i, i] = a
                    residual = jcope1[i, 0] - a*jcope0[i, 0]
                    F[i, -1] = np.float32(residual)
                F[_2N0, -1] = np.float32(1.0)
                # print(f'F:{F}F')
                return F.tocsr()

            def B_mat():
                B = sps.lil_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32))
                range_lat = 5 # TODO
                range_lon = 10 # TODO
                lat_halflife = 0.037
                lon_halflife = 0.064
                deg_per_mesh = 1/18
                theta = -np.arctan(1/3)
                for i in range(_N0):
                    grid0, grid1 = kurosio_grid_pooled[i]
                    for dy in np.arange(0, range_lat, 0.3):
                        for dx in np.arange(0, range_lon, 0.3):
                            # 重みの半減距離は度数で計算するため、度数に変換
                            dydeg =  dy*deg_per_mesh # dy2方向の度数 
                            dxdeg =  dx*deg_per_mesh # dx2方向の度数
                            w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))

                            # 黒潮方向dy2とdy2+90°方向dx2を計算
                            dy2 = int(dy*np.cos(theta) + dx*np.sin(theta))
                            dx2 = int(dx*np.cos(theta) - dy*np.sin(theta))
                            if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                            and not grid0<dy2 and not grid1<dx2:
                                idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                                # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                                if idx!=-1: 
                                    B[i, idx] = w
                                    B[i+_N0, idx] = w

                            dy2 = int(dy*np.cos(theta) - dx*np.sin(theta))
                            dx2 = int(-dx*np.cos(theta) - dy*np.sin(theta))
                            if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                            and not grid0<dy2 and not grid1<dx2:
                                idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                                # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                                if idx!=-1: 
                                    B[i, idx] = w
                                    B[i+_N0, idx] = w

                            dy2 = int(-dy*np.cos(theta) + dx*np.sin(theta))
                            dx2 = int(dx*np.cos(theta) + dy*np.sin(theta))
                            if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                            and not grid0<dy2 and not grid1<dx2:
                                idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                                # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                                if idx!=-1: 
                                    B[i, idx] = w
                                    B[i+_N0, idx] = w

                            dy2 = int(-dy*np.cos(theta) - dx*np.sin(theta))
                            dx2 = int(-dx*np.cos(theta) + dy*np.sin(theta))
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

            def H_mat(phi2):
                # min_value = 1/1e10
                H = sps.lil_matrix(np.zeros((_N+1, _2NM+1), dtype=np.float32))
                for i in range(_N):
                    H[i, i] = np.cos(phi2[i])
                    H[i, i+_N] = np.sin(phi2[i])
                H[-1, -1] = 1.0
                return H
                
            def R_mat(sigma2):

                R = sps.lil_matrix(np.zeros((_N+1, _N+1), dtype=np.float32))
                for i in range(_N):
                    R[i, i] = sigma2[i]
                return R.tocsr()

            def get_z(ais_cur2, dtidx, default=None):

                if len(ais_cur2)==0: 
                    print(f'Not exist data({dtidx}) in ais cur 2')
                    nan_data = [np.nan for _ in range(_N0)] + [1]
                    return np.array(nan_data) 

                ais_cur2_dt = kurosio_filter_pooled(ais_cur2, nan_map_pooled, is_pooled=True)[TF] # 時刻0のais data
                ais_cur12_dt = np.concatenate([ais_cur2_dt, [1]])
                ais_cur12_dt = ais_cur12_dt.reshape(_N0+1, 1)

                tf = np.isnan(ais_cur12_dt)
                if default==None:
                    ais_cur12_dt[tf] = np.nan 
                else:
                    ais_cur12_dt[tf] = default[tf]
                return sps.csr_matrix(ais_cur12_dt)

            def get_x(jcope_n, jcope_e, dtidx, default=np.nan):

                jcope_n_dt = kurosio_filter_pooled(jcope_n[dtidx], nan_map_pooled) # 時刻0のjcope data
                jcope_n_dt = jcope_n_dt[TF] # 時刻0のjcope data

                jcope_e_dt = kurosio_filter_pooled(jcope_e[dtidx], nan_map_pooled)[TF] # 時刻0のjcope data

                jcope_ne_dt = np.concatenate([jcope_n_dt, jcope_e_dt])
                jcope_ne_dt = np.concatenate([jcope_ne_dt, [1]])
                jcope_ne_dt = jcope_ne_dt.reshape(_2N0+1, 1)
                jcope_ne_dt[np.isnan(jcope_ne_dt)] = 0.0
                return sps.csr_matrix(jcope_ne_dt)


            # Setting Kalman Param
            pm.printline('Setting Kalman parameter')

            ## 初期の観測値と状態値
            z_csr = get_z(ais_cur2, dtidx)

            z = z_csr.toarray()
            nonZero = z!=0.0
            tf = (~np.isnan(z)) & (nonZero)
            tf_ravel = tf[:_N0].ravel()
            
            lambda2 = kurosio_filter_pooled(ais_lambda2, nan_map_pooled, is_pooled=True)[TF]
            indices = np.argsort(-lambda2)[0:N_lambda]

            tf_lambda2 = np.array([False]*len(lambda2))
            tf_lambda2[indices] = True
            tf_lambda2 = (~np.isnan(lambda2)) & (lambda2>Min_lambda) & (tf_lambda2)
            tf_ravel = (tf_ravel) & (tf_lambda2)
            tf = np.concatenate([tf_ravel, [True]])
            tf = tf.reshape(_N0+1, 1)
            tf2 = np.concatenate([tf_ravel, tf_ravel])
            tf2 = np.concatenate([tf2, [True]])

            phi2 = kurosio_filter_pooled(ais_phi2, nan_map_pooled, is_pooled=True)[TF]

            z = z[tf]
            lambda2 = lambda2[tf_ravel]
            phi2 = phi2[tf_ravel] 
            
            n = _N = int(len(z)-1)
            m = _M = 1 
            _NM = _N * _M
            _2NM = 2 * _NM

            ## H関数
            H = H_mat(phi2)

            ## 初期の推定値x
            jcope_n, jcope_e = jl.load_jcope_day(day)
            x = jcope = get_x(jcope_n, jcope_e, dtidx)
            print(f'jcope shape: {jcope_n[dtidx].shape}')
            assert np.sum(np.isnan(x.toarray()))==0

            ## F関数
            jcope0 = get_x(jcope_n, jcope_e, dtidx)
            jcope1 = get_x(jcope_n, jcope_e, dtidx+1)
            F = F_mat(jcope0.tolil(), jcope1.tolil())

            # ## 分散共分散行列
            # Q = sps.eye(_2N0+1, dtype=np.float32)*np.float32(Q_value) # システム誤差(jcope)
            # Q = Q.tolil()
            # Q[-1, -1] = 0.0
            # Q = Q.tocsr()

            # B = B_mat()

            # sigma2 = 1/(2*lambda2)
            # R = R_mat(sigma2) # 観測誤差(ais)

            # P = sps.csr_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32)) # 初期事後誤差共分散行列

            # I = sps.eye(_2NM+1, dtype=np.float32) # 初期事後誤差共分散行列

test()