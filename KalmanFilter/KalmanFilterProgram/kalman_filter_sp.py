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

from KalmanFilterProgram.Ais4v2ToCur import ais4
import KalmanFilterProgram.Ais4v2ToCur as atc
from utils import *
from utils.kalman_parameters import *
from utils.utils_needed_params import *
from KalmanFilterProgram.utils.kalman_funcs import *

def kalman_filter():
    if use_shipvar:
        print(f'use shipvar to calc ais cur')
    if use_ais_remove_bad_mmsi:
        print(f'use ais removed bad mmsi to calc ais cur')
        
    for Q_value in Q_values:
        save_dir = path_save+f"-Q{Q_value}" # データの保存先
        atc.svm.set_out_folder(save_dir) # 船毎の分散のリセット
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
        if use_ais_median:
            ais_keys = ['cur1_2', 'lambda1_2', 'phi1_2', 'cur2_2', 'lambda2_2', 'phi2_2']
        elif use_ais_remove_bad_mmsi:
            ais_keys = ['cur1', 'lambda1', 'psi1', 'cur2', 'lambda2', 'psi2']
        else:
            ais_keys = ['cur1', 'lambda1', 'phi1', 'cur2', 'lambda2', 'phi2']
            
        ais_outfiles = save_dir + r"\ais_files"
        if not use_ais_remove_bad_mmsi:
            AISLoader = atc.AISLoader
            al = AISLoader(year, month, ais_outfiles, pkl_path=path_ais)
            al.set_keys(ais_keys)
        else:
            from utils.ais_loader import AISLoader
            al = AISLoader(year, month)
            al.set_keys(ais_keys)
            al.load_path()

        # filteringの前準備
        pm.clear()
        pm.printline(f'Loading ais and jcope now')

        day = 1 
        dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)

        # AISの読み込み
        if use_shipvar:
            data = al.load_ais_dtidx(dtidx)
        elif use_ais_remove_bad_mmsi:
            data = al.load_ais_dtidx(dtidx)
        else:
            data = al.load_cur(dtidx)

        ais_cur1, ais_lambda1, ais_phi1, ais_cur2, ais_lambda2, ais_phi2\
            = [data[key] for key in ais_keys]
        cur2 = kurosio_filter_pooled(ais_cur2, nan_map_pooled, is_pooled=True)# 時刻0のais data
        print(f'ais shape: {ais_cur2.shape}')

        # データ数の設定
        n = _N = _N0 = len(cur2)
        m = _M = 1
        _NM = _N * _M
        _2NM = _2N0 = 2 * _NM
        print(f'Target data num: {_2NM}')



        # Setting Kalman Param
        pm.printline('Setting Kalman parameter')

        ## 初期の観測値と状態値
        z_csr = get_z(ais_cur2, dtidx, _N0)

        z = z_csr.toarray()
        nonZero = z!=0.0
        tf = (~np.isnan(z)) & (nonZero)
        tf_ravel = tf[:_N0].ravel()
        
        lambda2 = kurosio_filter_pooled(ais_lambda2, nan_map_pooled, is_pooled=True)
        indices = np.argsort(-lambda2)[0:N_lambda]

        tf_lambda2 = np.array([False]*len(lambda2))
        tf_lambda2[indices] = True
        tf_lambda2 = (~np.isnan(lambda2)) & (lambda2>Min_lambda) & (tf_lambda2)
        tf_ravel = (tf_ravel) & (tf_lambda2)
        tf = np.concatenate([tf_ravel, [True]])
        tf = tf.reshape(_N0+1, 1)
        tf2 = np.concatenate([tf_ravel, tf_ravel])
        tf2 = np.concatenate([tf2, [True]])

        phi2 = kurosio_filter_pooled(ais_phi2, nan_map_pooled, is_pooled=True)

        z = z[tf]
        lambda2 = lambda2[tf_ravel]
        phi2 = phi2[tf_ravel] 
        
        n = _N = int(len(z)-1)
        m = _M = 1 
        _NM = _N * _M
        _2NM = 2 * _NM

        ## H関数
        target_idxs = np.where(tf_ravel)[0]
        H = H_mat(phi2, target_idxs, _N, _N0)

        ## 初期の推定値x
        jcope_n, jcope_e = jl.load_jcope_day(day)
        x = jcope = get_x(jcope_n, jcope_e, dtidx, _2N0)
        print(f'jcope shape: {jcope_n[dtidx].shape}')
        assert np.sum(np.isnan(x.toarray()))==0

        ## F関数
        jcope0 = get_x(jcope_n, jcope_e, dtidx, _2N0)
        jcope1 = get_x(jcope_n, jcope_e, dtidx+1, _2N0)
        F = F_mat(jcope0.tolil(), jcope1.tolil(), _2N0)

        ## 分散共分散行列
        Q = sps.eye(_2N0+1, dtype=np.float32)*np.float32(Q_value) # システム誤差(jcope)
        Q = Q.tolil()
        Q[-1, -1] = 0.0
        Q = Q.tocsr()

        B = B_mat(_N0)
        assert np.sum(np.isnan(B.toarray()))==0

        sigma2 = 1/(2*lambda2)
        R = R_mat(sigma2, _N) # 観測誤差(ais)

        P = sps.csr_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32)) # 初期事後誤差共分散行列

        I = sps.eye(_2N0+1, dtype=np.float32) # 初期事後誤差共分散行列

        ## データの保存
        WHour = (1/2)**(1/8.5)
        r = 0
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

            for t in tqdm.tqdm(range(0,24)):
                if day==1 and t==0:
                    continue
                # データの読み込み
                dt = datetime.datetime(dt_year, dt_month, day, t, 0, 0)
                dtidx = date_to_dtidx(base_dt, dt)
                if use_shipvar:
                    data  = al.load_ais_dtidx(dtidx)
                elif use_ais_remove_bad_mmsi:
                    data  = al.load_ais_dtidx(dtidx)
                else:
                    data  = al.load_cur(dtidx)
                ais_cur1, ais_lambda1, ais_phi1, ais_cur2, ais_lambda2, ais_phi2 = [data[key] for key in ais_keys]
                ais_curs = [ais_cur1, ais_cur2]
                ais_lambdas = [ais_lambda1, ais_lambda2]
                ais_phis = [ais_phi1, ais_phi2]

                if t!=0:
                    jcope0 = get_x(jcope_n, jcope_e, dtidx-1, _2N0)
                    jcope1 = get_x(jcope_n, jcope_e, dtidx, _2N0)
                else:
                    jcope0 = get_x(p_jcope_n, p_jcope_e, dtidx-1, _2N0)
                    jcope1 = get_x(jcope_n, jcope_e, dtidx, _2N0)


                F = F_mat(jcope0, jcope1, _2N0)

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

                    z_csr = get_z(ais_cur, dtidx, _N0)
                    z = z_csr.toarray()
                    tf = (~np.isnan(z)) & (nonZero)
                    tf_ravel = tf[:_N0].ravel()
                    
                    pm.printline('Filtering step (Get lambda)')
                    lambda2 = kurosio_filter_pooled(ais_lambda, nan_map_pooled, is_pooled=True)
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
                    phi2 = kurosio_filter_pooled(ais_phi, nan_map_pooled, is_pooled=True)

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
                    H = H_mat(phi2, target_idxs, _N, _N0)

                    pm.printline('Filtering step (calc R)')
                    R = R_mat(sigma2, _N) # 観測誤差(ais)

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
                    jcope = get_x(jcope_n, jcope_e, dtidx, _2N0)
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
                    #pkl.dump(P.toarray(), open(f'{save_dir}/saverP{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    #pkl.dump(R.toarray(), open(f'{save_dir}/saverR{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    #pkl.dump(F.toarray(), open(f'{save_dir}/saverF{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    pkl.dump(H.toarray(), open(f'{save_dir}/saverH{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    #pkl.dump(K.toarray(), open(f'{save_dir}/saverK{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    pkl.dump(jcor.toarray(), open(f'{save_dir}/saverJCOPECur{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    pkl.dump((H@x).toarray(), open(f'{save_dir}/saverXCur{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    pkl.dump(tf, open(f'{save_dir}/saverTarget{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'wb'))
                    if use_shipvar:
                        atc.svm.save_info(f'{year}{month:02}{day:02}{t:02}-{target_name}')
                    p_jcope_n = jcope_n
                    p_jcope_e = jcope_e

        pm.printline('Finished KalmanFilter')
        logger.reset()