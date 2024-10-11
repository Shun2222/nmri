###
# Kalman Filter 
# フィルタ結果と船の推定偏流との誤差を計算し、船の信頼度を更新する
###
print('\rimport files now', end='')
import datetime
import numpy as np
import pandas as pd
from numpy.random import randn
import seaborn as sns
import pickle as pkl
import math
import os 
import os.path as osp
import re
import scipy as sp
import scipy.sparse as sps
from tqdm import tqdm

from KalmanFilterProgram.Ais4ToCur import ais4
import KalmanFilterProgram.Ais4ToCur as atc
from utils import *
from utils.kalman_parameters import *
from utils.utils_needed_params import *

target_point = np.array([105, 97], dtype=np.int64)
target_idx = kurosio_index_pooled[target_point[0]][target_point[1]]
def get_dammy_x(n):
    return np.zeros((n, 1))

def dammy_datas(no_data=True):
    ais_keys = ['cur1', 'lambda1', 'phi1', 'cur2', 'lambda2', 'phi2']
    dammies = {}

    dammy = np.nan * np.zeros(nan_map_pooled.shape)
    if not no_data:
        dammy[target_point[0]][target_point[1]] = -76 #-0.76
    dammies['cur1'] = dammy

    dammy = np.nan * np.zeros(nan_map_pooled.shape)
    if not no_data:
        dammy[target_point[0]][target_point[1]] = 26.53
    dammies['lambda1'] = dammy

    dammy = np.nan * np.zeros(nan_map_pooled.shape)
    if not no_data:
        dammy[target_point[0]][target_point[1]] = -1.95
    dammies['phi1'] = dammy
    dammy = np.nan * np.zeros(nan_map_pooled.shape)
    if not no_data:
        dammy[target_point[0]][target_point[1]] = -17 #-0.17
    dammies['cur2'] = dammy

    dammy = np.nan * np.zeros(nan_map_pooled.shape)
    if not no_data:
        dammy[target_point[0]][target_point[1]] = 375.42
    dammies['lambda2'] = dammy

    dammy = np.nan * np.zeros(nan_map_pooled.shape)
    if not no_data:
        dammy[target_point[0]][target_point[1]] = 2.77
    dammies['phi2'] = dammy

    return dammies


def kalman_filter():
    for Q_value in Q_values:
        save_dir = path_save+f"{Q_value}" # データの保存先
        pm.clear()

        # パラメータの設定
        year = dt_year = 2015
        month = dt_month = 9
        dt_day = 1
        n_day = nday_month(dt_month) 
        base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)


        logger.configure(save_dir, format_strs=['stdout', 'log', 'json'])
        dump_params()

        # データの読み込みクラスの設定
        pm.printline('Setting jcope loader')
        jl = JCOPELoader(year, month)
        jl.load_path(path_jcope)

        pm.printline('Setting ais loader')
        ais_keys = ['cur1', 'lambda1', 'phi1', 'cur2', 'lambda2', 'phi2']
        ais_outfiles = save_dir + r"\ais_files"
        al = atc.AISLoader(year, month, ais_outfiles, pkl_path=path_ais)
        al.set_keys(ais_keys)

        # filteringの前準備
        pm.clear()
        for r in range(1):
            pm.printline(f'Loading ais and jcope now (n_slice={r})')
            atc.svm.clear() # 船毎の分散のリセット

            day = 1 
            dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)

            # AISの読み込み
            #data  = al.load_cur(dtidx)
            data  = dammy_datas(no_data=False) 
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
                deg_per_mesh = 1/(36/pool_size)
                theta = -np.arctan(1/3)
                for i in range(_N0):
                    grid0, grid1 = kurosio_grid_pooled[i]
                    for dy in np.arange(0, range_lat, 0.3):
                        for dx in np.arange(0, range_lon, 0.3):
                            # 重みの半減距離は度数で計算するため、度数に変換
                            # dydeg =  dy*deg_per_mesh # dy2方向の度数 
                            # dxdeg =  dx*deg_per_mesh # dx2方向の度数
                            # w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))

                            # 以下、４つの方向でそれぞれ重みを計算し、行列Bに反映
                            # 黒潮方向dy2とdy2+90°方向dx2を計算
                            dy2 = int(dy*np.cos(theta) + dx*np.sin(theta))
                            dx2 = int(dx*np.cos(theta) - dy*np.sin(theta))
                            dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                            dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                            w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                            if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                            and not grid0<dy2 and not grid1<dx2:
                                idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                                # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                                if idx!=-1: 
                                    B[i, idx] = w
                                    B[i+_N0, idx] = w

                            dy2 = int(dy*np.cos(theta) - dx*np.sin(theta))
                            dx2 = int(-dx*np.cos(theta) - dy*np.sin(theta))
                            dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                            dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                            w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                            if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                            and not grid0<dy2 and not grid1<dx2:
                                idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                                # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                                if idx!=-1: 
                                    B[i, idx] = w
                                    B[i+_N0, idx] = w

                            dy2 = int(-dy*np.cos(theta) + dx*np.sin(theta))
                            dx2 = int(dx*np.cos(theta) + dy*np.sin(theta))
                            dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                            dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                            w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                            if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                            and not grid0<dy2 and not grid1<dx2:
                                idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                                # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                                if idx!=-1: 
                                    B[i, idx] = w
                                    B[i+_N0, idx] = w

                            dy2 = int(-dy*np.cos(theta) - dx*np.sin(theta))
                            dx2 = int(-dx*np.cos(theta) + dy*np.sin(theta))
                            dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                            dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                            w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
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

            def H_mat(phi, idxs):
                # min_value = 1/1e10
                H = sps.lil_matrix(np.zeros((_N+1, _2N0+1), dtype=np.float32))
                for i in range(_N):
                    idx = idxs[i]
                    H[i, idx] = np.cos(phi[i])
                    H[i, idx+_N0] = np.sin(phi[i])
                H[-1, -1] = 1.0
                length = H.shape[0] # これを入れるとerrorなくなる (配列の長さを計算して内部で保持してる？)
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
                return sps.csr_matrix(get_dammy_x(_2N0+1))

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
            target_idxs = np.where(tf_ravel)[0]
            H = H_mat(phi2, target_idxs)

            ## 初期の推定値x
            #jcope_n, jcope_e = jl.load_jcope_day(day)
            jcope_n = jcope_e = None
            x = jcope = get_x(jcope_n, jcope_e, dtidx)
            #print(f'jcope shape: {jcope_n[dtidx].shape}')
            assert np.sum(np.isnan(x.toarray()))==0

            ## F関数
            jcope0 = get_x(jcope_n, jcope_e, dtidx)
            jcope1 = get_x(jcope_n, jcope_e, dtidx+1)
            F = F_mat(jcope0.tolil(), jcope1.tolil())

            ## 分散共分散行列
            Q = sps.eye(_2N0+1, dtype=np.float32)*np.float32(Q_value) # システム誤差(jcope)
            Q = Q.tolil()
            Q[-1, -1] = 0.0
            Q = Q.tocsr()

            B = B_mat()

            sigma2 = 1/(2*lambda2)
            R = R_mat(sigma2) # 観測誤差(ais)

            P = sps.csr_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32)) # 初期事後誤差共分散行列

            I = sps.eye(_2N0+1, dtype=np.float32) # 初期事後誤差共分散行列

            ## データの保存
            WHour = (1/2)**(1/8.5)
            pkl.dump(x.toarray(), open(f'{save_dir}/saverX{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            pkl.dump(jcope.toarray(), open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            pkl.dump(P.toarray(), open(f'{save_dir}/saverP{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            pkl.dump(R.toarray(), open(f'{save_dir}/saverR{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            pkl.dump(F.toarray(), open(f'{save_dir}/saverF{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            pkl.dump(H.toarray(), open(f'{save_dir}/saverH{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            #jcope_lil = jcope.tolil()
            #jcope2 = jcope_lil[tf2].tocsr()
            jcor = H @ jcope
            pkl.dump(jcor.toarray(), open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}00-{r}.pkl', 'wb'))
            pkl.dump((H@x).toarray(), open(f'{save_dir}/saverXCur{year}{month:02}{day:02}00-{r}.pkl', 'wb'))

            # Filtering開始
            pm.printline('Start KalmanFilter')
            for day in range(1, n_day):
                dt = datetime.datetime(dt_year, dt_month, day)
                print(f'Filter dt:{dt}')
                pm.printline(f'Filtering {dt}')
                
                if day != 1:
                    jcope_n, jcope_e = jl.load_jcope_day(day)
                    jcope_n = jcope_e = None
                for t in tqdm.tqdm(range(0,24)):
                    if day==1 and t==0:
                        continue

                    # データの読み込み
                    dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
                    dtidx = date_to_dtidx(base_dt, dt)
                    #data = al.load_cur(dtidx)
                    if t==1:
                        data = dammy_datas(no_data=False)
                    else:
                        data = dammy_datas(no_data=True)
                    ais_cur1, ais_lambda1, ais_phi1, ais_cur2, ais_lambda2, ais_phi2 = [data[key] for key in ais_keys]
                    ais_curs = [ais_cur1, ais_cur2]
                    ais_lambdas = [ais_lambda1, ais_lambda2]
                    ais_phis = [ais_phi1, ais_phi2]
                    
                    if t!=0:
                        jcope0 = get_x(jcope_n, jcope_e, dtidx-1)
                        jcope1 = get_x(jcope_n, jcope_e, dtidx)
                    else:
                        jcope0 = get_x(p_jcope_n, p_jcope_e, dtidx-1)
                        jcope1 = get_x(jcope_n, jcope_e, dtidx)

                    F = F_mat(jcope0, jcope1)

                    # 予測ステップ
                    pm.printline('Predicting step (update x)')
                    #x = x
                    x = F @ x #x k|k-1

                    pm.printline('Predicting step (update P-)')
                    P = F @ P @ F.T + B @ Q @ B.T 
                    #pkl.dump(P, open(f'{save_dir}/saverPminus{year}{month:02}{day:02}{t:02}.pkl', 'wb'))
                    
                    # フィルタリングステップ
                    ## v1, v2それぞれでフィルタリング
                    for ais_cur, ais_lambda, ais_phi, target_name in zip(ais_curs, ais_lambdas, ais_phis, ['v1', 'v2']):
                        pm.printline('Filtering step (Get z)')

                        z_csr = get_z(ais_cur, dtidx)
                        z = z_csr.toarray()
                        tf = (~np.isnan(z)) & (nonZero)
                        tf_ravel = tf[:_N0].ravel()
                        
                        pm.printline('Filtering step (Get lambda)')
                        lambda2 = kurosio_filter_pooled(ais_lambda, nan_map_pooled, is_pooled=True)[TF]
                        indices = np.argsort(-lambda2)[0:N_lambda]

                        tf_lambda2 = np.array([False]*len(lambda2))
                        tf_lambda2[indices] = True
                        tf_lambda2 = (~np.isnan(lambda2)) & (lambda2>Min_lambda) & (tf_lambda2)
                        tf_ravel = (tf_ravel) & (tf_lambda2)
                        tf = np.concatenate([tf_ravel, [True]])
                        tf = tf.reshape(_N0+1, 1)
                        tf2 = np.concatenate([tf_ravel, tf_ravel])
                        tf2 = np.concatenate([tf2, [True]])

                        pm.printline('Filtering step (Get phi)')
                        phi2 = kurosio_filter_pooled(ais_phi, nan_map_pooled, is_pooled=True)[TF]

                        z = z[tf]
                        z = sps.csr_matrix(z.reshape(len(z), 1))
                        lambda2 = lambda2[tf_ravel]
                        phi2 = phi2[tf_ravel] 
                        sigma2 = 1/(2*lambda2)
                        
                        pm.printline('Filtering step (update parameter)')
                        n = _N = int(z.shape[0]-1)
                        m = _M = 1 
                        _NM = _N * _M
                        _2NM = 2 * _NM
                        print(f'New _N = {_N}')
                        
                        #kf.R = R_mat(sigma1, sigma2) # 観測誤差(ais)
                        pm.printline('Filtering step (calc H)')
                        target_idxs = np.where(tf_ravel)[0]
                        H = H_mat(phi2, target_idxs)

                        pm.printline('Filtering step (calc R)')
                        R = R_mat(sigma2) # 観測誤差(ais)

                        # データにnanがないかのチェック
                        assert np.sum(np.isnan(lambda2))==0
                        assert np.sum(np.isnan(sigma2))==0
                        assert np.sum(np.isnan(phi2))==0
                        assert np.sum(np.isnan(z.toarray()))==0
                        assert np.sum(np.isnan(x.toarray()))==0
                        assert np.sum(np.isnan(P.toarray()))==0
                        assert np.sum(np.isnan(H.toarray()))==0
                        assert np.sum(np.isnan(R.toarray()))==0


                        # kf.update(z)
                        pm.printline('Filtering step (calc S)')
                        PHT = P @ H.T
                        S = H @ PHT + R
                        assert np.sum(np.isnan(S.toarray()))==0

                        inv = 'pinv'
                        pm.printline('Filtering step (calc K)')
                        SI_np = np.linalg.pinv(S.T.toarray())
                        SI = sps.csr_matrix(SI_np)
                        K = PHT @ SI

                        pm.printline('Filtering step (filetering x)')
                        #x_lil = x.tolil()
                        #x2 = x_lil[tf2].tocsr()
                        y = z - H @ x
                        x = x + K @ y # x k|k
                        #x_lil[tf2] = x2
                        #x = x_lil.tocsr()

                        pm.printline('Filtering step (update P)')
                        print(f'I {I.shape}')
                        print(f'K {K.shape}')
                        print(f'H {H.shape}')
                        I_KH = I - K @ H
                        # TODO ちょっと知ってるのと違う，式的にRを写像してPに加えてる，Rの誤差も考慮するようにしてる?
                        #P_np = P.toarray()
                        #Ptf = sps.csr_matrix(P_np[tf2.ravel()])
                        P = I_KH @ P @ I_KH.T + K @ R @ K.T
                        #Ptf_np = Ptf.toarray()
                        #Ptf_np_prev = P_np[tf2.ravel()]
                        #for i in range(Ptf_np.shape[0]):
                        #    Ptf_np_prev[i][tf2.ravel()] = Ptf_np[i] # TODO
                        #P_np[tf2.ravel()] = Ptf_np_prev
                        #P = sps.csr_matrix(P_np)
                    
                        pm.printline('Calc jcor')
                        jcope = get_x(jcope_n, jcope_e, dtidx)
                        #jcope_lil = jcope.tolil()
                        #jcope2 = jcope_lil[tf2].tocsr()
                        jcor = H @ jcope

                        def mean_diff(x, y):
                            a = x-y
                            diff = np.mean(np.abs(a))
                            return diff

                        pm.clear()
                        logger.record_tabular(f"dtidx", dtidx)
                        logger.record_tabular(f"SI", inv)
                        logger.record_tabular(f"Available AIS", np.sum(tf_ravel))
                        logger.dump_tabular()

                        # 保存
                        pm.printline('Saving datas')
                        pkl.dump(x.toarray(), open(f'{save_dir}/saverX{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(z.toarray(), open(f'{save_dir}/saverZ{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(jcope.toarray(), open(f'{save_dir}/saverJCOPE{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(P.toarray(), open(f'{save_dir}/saverP{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(R.toarray(), open(f'{save_dir}/saverR{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(F.toarray(), open(f'{save_dir}/saverF{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(H.toarray(), open(f'{save_dir}/saverH{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(K.toarray(), open(f'{save_dir}/saverK{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(jcor.toarray(), open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump((H@x).toarray(), open(f'{save_dir}/saverXCur{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        pkl.dump(tf, open(f'{save_dir}/saverTarget{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                        atc.svm.save_info(f'{year}{month:02}{day:02}{t:02}-{target_name}')

                        p_jcope_n = jcope_n
                        p_jcope_e = jcope_e
            pm.printline('Finished KalmanFilter')
        logger.reset()