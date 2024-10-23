import pickle as pkl
import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import re
import os 
import os.path as osp
import pandas as pd
import seaborn as sns
import datetime 
import tqdm

from utils import * 
import KalmanFilterProgram.Ais4ToCur as atc
from utils.analysis_parameters import *
from utils.utils_needed_params import *
from utils.utils_visualization import GifMaker


dt_year = year = 2015
dt_month = month = 9
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True
n_day = MAX_DAY

done_ship = []
s = 'v2' #v2 or 0
kalman_keys = ['X', 'JCOPE', 'Z', 'Target']
if not use_ais_removed_bad_mmsi:
    ais_keys = ['n', 'e']
else:
    ais_keys = ['N', 'E', 'lambda1']


def plot_mappedData(kl, al, target_day, target_hour, filter_latlon=None, filter_range=30, filter=False):
    #for s in range(0, 1):
    for s in ['v1', 'v2']:
        kalman_n = []
        kalman_e = []
        day = target_day
        print(f's:{s}, day:{day}, hour:{target_hour}')
        # TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']
        if is_light:
            TF = np.array([True] * 3789)
        else:
            TF = np.array([True] * 9808)

        hour = target_hour     
        data = kl.load_kalmanLog_day_hour(day, hour, s, keys=kalman_keys)
        dt = datetime.datetime(year, month, day, hour, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        if dtidx==0:
            continue
        
        n_data = int((len(data['X'])-1)/2)
        
        kalman_n = data['X'][:n_data]
        kalman_e = data['X'][n_data:-1]
        jcope_n = data['JCOPE'][:n_data]
        jcope_e = data['JCOPE'][n_data:-1]
        
        if is_light:
            n_map = kurosio_vec_to_map_pooled(kalman_n, nan_map_pooled) * nan_map_pooled
            e_map = kurosio_vec_to_map_pooled(kalman_e, nan_map_pooled) * nan_map_pooled
            jn_map = kurosio_vec_to_map_pooled(jcope_n, nan_map_pooled) * nan_map_pooled
            je_map = kurosio_vec_to_map_pooled(jcope_e, nan_map_pooled) * nan_map_pooled
            kjn_map = kurosio_vec_to_map_pooled(np.abs(kalman_n-jcope_n), nan_map_pooled) * nan_map_pooled
            kje_map = kurosio_vec_to_map_pooled(np.abs(kalman_e-jcope_e), nan_map_pooled) * nan_map_pooled
            point_map = np.zeros(nan_map_pooled.shape) 
            point_map[105][97] = 1
            point_map *= nan_map_pooled
        else:
            n_map = kurosio_vec_to_map(kalman_n, nan_map) * nan_map
            e_map = kurosio_vec_to_map(kalman_e, nan_map) * nan_map
            jn_map = kurosio_vec_to_map(jcope_n, nan_map) * nan_map
            je_map = kurosio_vec_to_map(jcope_e, nan_map) * nan_map
            
        
        data = al.load_cur(dtidx)
        an_map, ae_map = [data[key] for key in ais_keys]

        if filter:
            lat00 = filter_latlon[0][0]
            lat11 = filter_latlon[0][1]
            lon0 = filter_latlon[1][0]
            lon1 = filter_latlon[1][1]
            dif = filter_range

            n_map = n_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            e_map = e_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            jn_map = jn_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            je_map = je_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            kjn_map = kjn_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            kje_map = kje_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            an_map = an_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            ae_map = ae_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
            point_map = point_map[lat00-dif:lat11+dif].T[lon0-dif:lon1+dif].T
        else:
            point_map = n_map

        print('----------max----------')
        print(f'Max value')
        print(f'n {np.nanmax(n_map)}')
        print(f'e {np.nanmax(e_map)}')
        print(f'jcope n {np.nanmax(jn_map)}')
        print(f'jcope e {np.nanmax(je_map)}')
        print(f'ais n {np.nanmax(an_map)}')
        print(f'ais e {np.nanmax(ae_map)}')
        print('--------------------\n')

        print('----------min----------')
        print(f'Min value')
        print(f'n {np.nanmin(n_map)}')
        print(f'e {np.nanmin(e_map)}')
        print(f'jcope n {np.nanmin(jn_map)}')
        print(f'jcope e {np.nanmin(je_map)}')
        print(f'ais n {np.nanmin(an_map)}')
        print(f'ais e {np.nanmin(ae_map)}')
        print('--------------------\n')

        df_n_map = pd.DataFrame(n_map)
        df_e_map = pd.DataFrame(e_map)    
        df_jn_map = pd.DataFrame(jn_map)
        df_je_map = pd.DataFrame(je_map)
        df_kjn_map = pd.DataFrame(kjn_map)
        df_kje_map = pd.DataFrame(kje_map)
        df_ae_map = pd.DataFrame(an_map)
        df_an_map = pd.DataFrame(ae_map)
        df_nan_map = pd.DataFrame(nan_map_pooled)
        
        path = osp.join(path_log, f'kalman_n{year}{month:02}{day:02}{hour:02}.csv')
        df_n_map.to_csv(path, index=False, header=False)
        path = osp.join(path_log, f'jcope_n{year}{month:02}{day:02}{hour:02}.csv')
        df_jn_map.to_csv(path, index=False, header=False)
        path = osp.join(path_log, f'diff_kalman_jcope_n{year}{month:02}{day:02}{hour:02}.csv')
        df_kjn_map.to_csv(path, index=False, header=False)
        path = osp.join(path_log, f'ais_n{year}{month:02}{day:02}{hour:02}.csv')
        df_an_map.to_csv(path, index=False, header=False)
        # np.savetxt(path, kalman_n_map, delimiter=',', fmt='%f')
        
        path = osp.join(path_log, f'kalman_e{year}{month:02}{day:02}{hour:02}.csv')
        df_e_map.to_csv(path, index=False, header=False)
        path = osp.join(path_log, f'jcope_e{year}{month:02}{day:02}{hour:02}.csv')
        df_je_map.to_csv(path, index=False, header=False)
        path = osp.join(path_log, f'diff_kalman_jcope_e{year}{month:02}{day:02}{hour:02}.csv')
        df_kje_map.to_csv(path, index=False, header=False)
        path = osp.join(path_log, f'ais_e{year}{month:02}{day:02}{hour:02}.csv')
        df_ae_map.to_csv(path, index=False, header=False)
        # np.savetxt(path, kalman_e_map, delimiter=',', fmt='%f')


        path = osp.join(path_log, f'nan_map.csv')
        df_nan_map.to_csv(path, index=False, header=False)

        max_value = 3 
        min_value = -3

        plt.figure(figsize=(24, 24))
        plt.title(f'point {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(point_map, cbar=True)
        path = f'point_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(path_log, path)
        plt.savefig(path)
        plt.close()

        data = np.concatenate([an_map, jn_map])
        data = np.concatenate([data, n_map])
        plt.figure(figsize=(24, 24))
        plt.title(f'N {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'N_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(path_log, path)
        plt.savefig(path)
        plt.close()

        data[data<min_value] = np.nan
        data[data>max_value] = np.nan
        plt.figure(figsize=(24, 24))
        plt.title(f'N Filtered by {min_value}<=data<={max_value} {year}{month:02}{day:02}{hour:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'N_Filtered_{year}{month:02}{day:02}{hour:02}-{s}.png'
        path = osp.join(path_log, path)
        plt.savefig(path)
        plt.close()

        data = np.concatenate([ae_map, je_map])
        data = np.concatenate([data, e_map])
        plt.figure(figsize=(24, 24))
        plt.title(f'E {year}{month:02}{day:02}{hour:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'E_{year}{month:02}{day:02}{hour:02}-{s}.png'
        path = osp.join(path_log, path)
        plt.savefig(path)
        plt.close()

        data[data<min_value] = np.nan
        data[data>max_value] = np.nan
        plt.figure(figsize=(24, 24))
        plt.title(f'E Filtered by {min_value}<=data<={max_value} {year}{month:02}{day:02}{hour:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'E_Filtered_{year}{month:02}{day:02}{hour:02}-{s}.png'
        path = osp.join(path_log, path)
        plt.savefig(path)
        plt.close()

        data = np.concatenate([kjn_map, kje_map])
        plt.figure(figsize=(24, 24))
        plt.title(f'NE {year}{month:02}{day:02}{hour:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'kalman_jcope_NE_{year}{month:02}{day:02}{hour:02}-{s}.png'
        path = osp.join(path_log, path)
        plt.savefig(path)
        plt.close()
    print(f"Finished saving mapped date.")
    return n_map, e_map, jn_map, je_map, kjn_map, kje_map, an_map, ae_map, point_map

def diff_results2(kl):
    if is_light:
        TFs = None
        n_data = -1
    else:
        TFs = [np.array([True]*9808)]
        n_data = 9808
    #TFs = [kl.load_kalmanLog_day(1, s, keys=['TF'])['TF'] for s in range(14)]

    files = os.listdir(path_ship)
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))]

    df_aj = []
    df_as = []
    df_ak = []
    df_sj = []
    df_kj = []
    df_ks = []
    dtidxs = []
    grids = []
    diff_ship = {}
    for target_ship in target_ships:
        diff_ship[target_ship] = [[], [], 0]

    for day in range(1, n_day+1):
    #for day in range(23, 23+1):
        dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
        print(dt)
        
        for target_ship in target_ships:
            available_count = 0
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

            diffs = []
            for i in range(len(shipLog)):
                time = shipLog['DtIdx'].values[i]
                grid0 = shipLog['Grid0'].values[i]
                grid1 = shipLog['Grid1'].values[i]
                curN = shipLog['CurN'].values[i]
                curE = shipLog['CurE'].values[i]
                dtidxs.append(time)
                grids.append([grid0, grid1])
                dtidx = time

                if is_light:
                    grid0 = int(grid0/pool_size)
                    grid1 = int(grid1/pool_size)
                    # if not kurosio_pooled(grid0, grid1):
                    if not kurosio_map_tf_pooled[grid0][grid1]:
                        print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                        continue
                    idx = int(kurosio_index_pooled[grid0][grid1])
                else:
                    # if not kurosio(grid0, grid1):
                    if not kurosio_map_tf[grid0][grid1]:
                        print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                        continue
                    idx = int(kurosio_index[grid0][grid1])

                if dtidx<0:
                    print(f'Out of dtidx ( dtidx = {dtidx}) ).')
                    continue

                if idx==-1:
                    print(f'Out of idx ( grid = ({grid0},{grid1}) ).')
                    continue


                dt = dtidx_to_date(base_dt, int(dtidx))
                hour = dt.hour
                if day>MAX_DAY :
                    continue
                if day==MAX_DAY and hour>MAX_HOUR:
                    continue
                    
                if day==1 and hour==0:
                    continue
                print(day)
                print(hour)
                data = kl.load_kalmanLog_day_hour(day, hour, s, keys=['X', 'JCOPE'])
                if data:
                    if n_data == -1:
                        n_data = int((data['X'][1].shape[0] - 1)/2)
                        TFs = [np.array([True]*n_data)]


                    kfidx = get_kfidx(TFs[0], idx)
                    if kfidx==-1:
                        continue
                    kalman_cur1 = data['X'][kfidx]
                    kalman_cur2 = data['X'][kfidx+n_data]

                    jcope_cur1 = data['JCOPE'][kfidx]
                    jcope_cur2 = data['JCOPE'][kfidx+n_data]

                    hour = dtidx_to_date(base_dt, int(dtidx)).hour
                    path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}l.csv"
                    dataN = pd.read_csv(path, encoding="cp932", header=None)
                    ais_cur1 = dataN.values[grid0][grid1]
                    
                    path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}Y.csv"
                    dataE = pd.read_csv(path, encoding="cp932", header=None)
                    ais_cur2 = dataE.values[grid0][grid1]

                    path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}Lambda1.csv"
                    lambda1 = pd.read_csv(path, encoding="cp932", header=None)
                    lambda1 = lambda1.values[grid0][grid1]
                    
                    if lambda1 > 10 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                    #if ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                        df_aj.append([ais_cur1-jcope_cur1, ais_cur2-jcope_cur2])
                        df_as.append([ais_cur1-curN, ais_cur2-curE])
                        df_ak.append([ais_cur1-kalman_cur1, ais_cur2-kalman_cur2])
                    df_sj.append([jcope_cur1-curN, jcope_cur2-curE])    
                    df_kj.append([jcope_cur1-kalman_cur1, jcope_cur2-kalman_cur2])    
                    df_ks.append([kalman_cur1-curN, kalman_cur2-curE]) 
                    diffs.append([kalman_cur1-curN, kalman_cur2-curE])
                    available_count += 1 

                if available_count>0:
                    diffs = np.array(diffs)
                    diffs = diffs.reshape(diffs.shape[0], diffs.shape[1])
                    diff_N = diffs.T[0]
                    diff_E = diffs.T[1]
                    diff_ship[target_ship][0] += diff_N.tolist()
                    diff_ship[target_ship][1] += diff_E.tolist()
                    diff_ship[target_ship][2] += available_count
                

    print(f'aj: {np.mean(np.abs(df_aj))}, ndata={len(df_aj)}')
    print(f'ak: {np.mean(np.abs(df_ak))}, ndata={len(df_ak)}')
    print(f'as: {np.mean(np.abs(df_as))}, ndata={len(df_as)}')
    print(f'kj: {np.mean(np.abs(df_kj))}, ndata={len(df_kj)}')
    print(f'sj: {np.mean(np.abs(df_sj))}, ndata={len(df_sj)}')
    print(f'ks: {np.mean(np.abs(df_ks))}, ndata={len(df_ks)}')
    for target_ship in target_ships:
        a = np.mean(diff_ship[target_ship][0])
        b = np.mean(diff_ship[target_ship][1])
        c = diff_ship[target_ship][2]
        print(f'{target_ship}: {round(np.mean(a), 2)}, {round(np.mean(b), 2)}, {c}')
    return ['ais-jcope', 'ais-kalman', 'ais-ship', 'kalman-jcope', 'ship-jcope', 'kalman-ship'],\
           [np.mean(np.abs(df_aj)), np.mean(np.abs(df_ak)),np.mean(np.abs(df_as)),np.mean(np.abs(df_kj)),np.mean(np.abs(df_sj)),np.mean(np.abs(df_ks)),],\
           [len(df_aj), len(df_ak),len(df_as),len(df_kj),len(df_sj),len(df_ks)]

def diff_results(kl, al):
    data = kl.load_kalmanLog_day(1, s, keys=['X'])

    files = os.listdir(path_ship)
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))]

    df_aj = []
    df_as = []
    df_ak = []
    df_sj = []
    df_kj = []
    df_ks = []
    dtidxs = []
    grids = []
    diff_ship = {}
    for target_ship in target_ships:
        diff_ship[target_ship] = [[], [], 0]

    for day in range(1, n_day+1):
    #for day in range(23, 23+1):
        dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
        print(dt)
        
        for target_ship in target_ships:
            available_count = 0
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

            diffs = []
            for i in range(len(shipLog)):
                time = shipLog['DtIdx'].values[i]
                grid0 = shipLog['Grid0'].values[i]
                grid1 = shipLog['Grid1'].values[i]
                curN = shipLog['CurN'].values[i]
                curE = shipLog['CurE'].values[i]
                dtidxs.append(time)
                grids.append([grid0, grid1])
                dtidx = time

                if is_light:
                    grid0 = int(grid0/pool_size)
                    grid1 = int(grid1/pool_size)
                    # if not kurosio_pooled(grid0, grid1):
                    if not kurosio_map_tf_pooled[grid0][grid1]:
                        print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                        continue
                    idx = int(kurosio_index_pooled[grid0][grid1])
                else:
                    # if not kurosio(grid0, grid1):
                    if not kurosio_map_tf[grid0][grid1]:
                        print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                        continue
                    idx = int(kurosio_index[grid0][grid1])

                if dtidx<0:
                    print(f'Out of dtidx ( dtidx = {dtidx}) ).')
                    continue

                if idx==-1:
                    print(f'Out of idx ( grid = ({grid0},{grid1}) ).')
                    continue


                data = kl.load_kalmanLog_day(day, s, keys=['X', 'JCOPE'])
                if n_data == -1:
                    n_data = int((data['X'][1].shape[0] - 1)/2)
                    TFs = [np.array([True]*n_data)]

                #if not dtidx in data['X'].keys() and day!=1:
                if not dtidx in data['X'].keys():
                    if day!=1:
                        data = kl.load_kalmanLog_day(day-1, s, keys=['X', 'JCOPE'])
                    if not dtidx in data['X'].keys():
                        print(f'There is no kalman data (dtidx = {dtidx}).')
                        continue

                kfidx = get_kfidx(TFs[0], idx)
                kalman_cur1 = data['X'][dtidx][kfidx]
                kalman_cur2 = data['X'][dtidx][kfidx+n_data]

                jcope_cur1 = data['JCOPE'][dtidx][kfidx]
                jcope_cur2 = data['JCOPE'][dtidx][kfidx+n_data]

                dtidx2 = dtidx
                ais_data = al.load_ais_dtidx(dtidx2, keys=["N", "E", "lambda1"], use_pool=True)
                #print(f"dtidx: {dtidx2}")
                if type(ais_data["N"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    dataN = ais_data["N"]
                    #print(f"N shape: {dataN.shape}")
                    ais_cur1 = dataN[grid0][grid1]
                    
                    dataE = ais_data["E"]
                    ais_cur2 = dataE[grid0][grid1]

                    lambda1 = ais_data["lambda1"]
                    lambda1 = lambda1[grid0][grid1]
                    
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                    elif not lambda1 > 10:
                        print(f"lambda1 is low {lambda1}, cur: {ais_cur1}, {ais_cur2}")

                    #if lambda1 > 10 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                    if ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                        df_aj.append([ais_cur1-jcope_cur1, ais_cur2-jcope_cur2])
                        df_as.append([ais_cur1-curN, ais_cur2-curE])
                        df_ak.append([ais_cur1-kalman_cur1, ais_cur2-kalman_cur2])
                    df_sj.append([jcope_cur1-curN, jcope_cur2-curE])    
                    df_kj.append([jcope_cur1-kalman_cur1, jcope_cur2-kalman_cur2])    
                    df_ks.append([kalman_cur1-curN, kalman_cur2-curE]) 
                    diffs.append([kalman_cur1-curN, kalman_cur2-curE])
                    available_count += 1 

            if available_count>0:
                diffs = np.array(diffs)
                diffs = diffs.reshape(diffs.shape[0], diffs.shape[1])
                diff_N = diffs.T[0]
                diff_E = diffs.T[1]
                diff_ship[target_ship][0] += diff_N.tolist()
                diff_ship[target_ship][1] += diff_E.tolist()
                diff_ship[target_ship][2] += available_count
                

    print(f'aj: {np.mean(np.abs(df_aj))}, ndata={len(df_aj)}')
    print(f'ak: {np.mean(np.abs(df_ak))}, ndata={len(df_ak)}')
    print(f'as: {np.mean(np.abs(df_as))}, ndata={len(df_as)}')
    print(f'kj: {np.mean(np.abs(df_kj))}, ndata={len(df_kj)}')
    print(f'sj: {np.mean(np.abs(df_sj))}, ndata={len(df_sj)}')
    print(f'ks: {np.mean(np.abs(df_ks))}, ndata={len(df_ks)}')
    for target_ship in target_ships:
        a = np.mean(diff_ship[target_ship][0])
        b = np.mean(diff_ship[target_ship][1])
        c = diff_ship[target_ship][2]
        print(f'{target_ship}: {round(np.mean(a), 2)}, {round(np.mean(b), 2)}, {c}')
    return ['ais-jcope', 'ais-kalman', 'ais-ship', 'kalman-jcope', 'ship-jcope', 'kalman-ship'],\
           [np.mean(np.abs(df_aj)), np.mean(np.abs(df_ak)),np.mean(np.abs(df_as)),np.mean(np.abs(df_kj)),np.mean(np.abs(df_sj)),np.mean(np.abs(df_ks)),],\
           [len(df_aj), len(df_ak),len(df_as),len(df_kj),len(df_sj),len(df_ks)]

def analysis(plot=True, diff=True, save_gif=True, save_point_graph=True):
    kl = KalmanLogLoader(2015, 9, 20)
    kl.set_path(path_log)

    if not use_ais_removed_bad_mmsi:
        ais_outfiles = osp.join(path_ais, 'ais_files')
        AISLoader = atc.AISLoader
        al = AISLoader(year, month, ais_outfiles, pkl_path=path_ais)
        al.set_keys(ais_keys)
    else:
        from utils.ais_loader import AISLoader
        print(f"AIS is loaded from  cur clacled by removed mad mmsi")
        al = AISLoader(year, month)
        al.set_keys(ais_keys)
        al.load_path()

    eval_ais(kl, al)
    return

    keys = ['kalman-n', 'kalman-e', 'jcope-n', 'jcope-e', 'diff-kalman-jcope-n', 'diff-kalman-jcope-e', 'ais-n', 'ais-e']
    if plot:
        m_day = 2 if MAX_DAY==1 else MAX_DAY
        for day in range(1, MAX_DAY+1):
            # lat00 = int(map_pooled_size[0]-kurosio_latidx_range1[0]/pool_size)
            # lat11 = int(map_pooled_size[0]-kurosio_latidx_range2[1]/pool_size)
            # lon0 = int(map_pooled_size[1]-kurosio_lonidx_range[0]/pool_size)
            # lon1 = int(map_pooled_size[1]-kurosio_lonidx_range[1]/pool_size)
            #fileter_latlon = np.array([[lat00, lat11], [lon0, lon1]], dtype=np.int64)
            filter_latlon = np.array([[0, 0], [0, 0]], dtype=np.int64) #可視化して決める
            all_datas = {}
            for key in keys:
                all_datas[key] = []
            max_hour = MAX_HOUR if day==MAX_DAY else 24
            for hour in range(1, max_hour):
                n_map, e_map, jn_map, je_map, kjn_map, kje_map, an_map, ae_map, point_map = plot_mappedData(kl, al, day, hour, filter_latlon, filter=False)
                datas = [n_map, e_map, jn_map, je_map, kjn_map, kje_map, an_map, ae_map]
                for i in range(len(datas)):
                    all_datas[keys[i]].append(datas[i])
        if save_gif:
            for key in keys:
                gif_maker = GifMaker()

                datas  =  all_datas[key]
                titles2 = [f"({key}{year}{month:02}{DAY:02}{h-1:02})" for h in range(1, MAX_HOUR)]
                gif_maker.add_datas(datas)
                fname = f'{key}-test'
                gif_maker.make(titles=titles2, folder=path_log, file_name=fname)
                gif_maker.reset()
        if save_point_graph:
            for key in keys:
                data_np = np.array(all_datas[key])
                y = data_np[:, 105, 97]
                plt.plot(np.arange(len(y)), y)
                path = osp.join(path_log, 'point_graph.png')
                plt.savefig(path)

                y = data_np[0, 105-20:105:20, 97]
                plt.plot(np.arange(len(y)), y)
                path = osp.join(path_log, 'point_yaxis_graph.png')
                plt.savefig(path)

                y = data_np[0, 105, 97-20:90+20]
                plt.plot(np.arange(len(y)), y)
                path = osp.join(path_log, 'point_yaxis_graph.png')
                plt.savefig(path)
            
    if diff:
        diff_results(kl, al)

def eval_ais(kl, al):
    # 船のpathの設定
    files = os.listdir(path_ship)
    # path内にあるファイルの一覧取得
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))] 

    # aisと船の偏流値の差の大きさ(|ais-ship|)
    df_as = {   "Dtidx":[],
                "ShipName":[],
                "N":[],
                "E":[],
            }
    df_as_grid = [np.zeros(nan_map_pooled.shape) for _ in range(2)]
    count_grid = np.zeros(nan_map_pooled.shape)
    dtidxs = []
    grids = []
    diff_ship = {}
    for target_ship in target_ships:
        diff_ship[target_ship] = [[], [], 0]

    for day in range(1, n_day+1):
    #for day in range(23, 23+1):
        dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
        print(dt)
        
        for target_ship in target_ships:
            available_count = 0
            print(f'{target_ship} {dt_month:02}{day:02}')
            # 無視する船はスキップ
            if target_ship in done_ship:
                print(f'skip {target_ship}')
                continue

            # 船のログの読み込み
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

            diffs = []
            for i in range(len(shipLog)):
                # 船のログの格納
                time = shipLog['DtIdx'].values[i]
                grid0 = shipLog['Grid0'].values[i]
                grid1 = shipLog['Grid1'].values[i]
                curN = shipLog['CurN'].values[i]
                curE = shipLog['CurE'].values[i]
                dtidxs.append(time)
                grids.append([grid0, grid1])
                dtidx = time

                # グリッド座標の計算
                grid0 = int(grid0/pool_size)
                grid1 = int(grid1/pool_size)
                # if not kurosio_pooled(grid0, grid1):

                # エリア内にいるかのチェック
                if not kurosio_map_tf_pooled[grid0][grid1]:
                    print(f'Not Kurosio area ( grid = ({grid0},{grid1}) ).')
                    continue

                # エリア内かつ海であるかのチェック
                idx = int(kurosio_index_pooled[grid0][grid1])
                if idx==-1:
                    print(f'Out of idx ( grid = ({grid0},{grid1}) ).')
                    continue

                # 時間のチェック(下限)
                if dtidx<0:
                    print(f'Out of dtidx ( dtidx = {dtidx}) ).')
                    continue

                # 時間のチェック(上限)
                dt = dtidx_to_date(base_dt, int(dtidx))
                hour = dt.hour
                if day>MAX_DAY :
                    continue
                if day==MAX_DAY and hour>MAX_HOUR:
                    continue
                # 例外の時間
                if day==1 and hour==0:
                    continue

                dtidx2 = dtidx
                ais_data = al.load_ais_dtidx(dtidx2, keys=["N", "E", "lambda1"], use_pool=True)
                #print(f"dtidx: {dtidx2}")
                if type(ais_data["N"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    dataN = ais_data["N"]
                    #print(f"N shape: {dataN.shape}")
                    ais_cur1 = dataN[grid0][grid1]
                    
                    dataE = ais_data["E"]
                    ais_cur2 = dataE[grid0][grid1]

                    lambda1 = ais_data["lambda1"]
                    lambda1 = lambda1[grid0][grid1]
                    
                    # nanでなく，固有値が一定以上の場合で比較する
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                        continue
                    #elif not lambda1 > 10:
                    #    print(f"lambda1 is low {lambda1}, cur: {ais_cur1}, {ais_cur2}")
                    #    continue
                    else:
                        df_as["Dtidx"].append(dtidx)
                        df_as["ShipName"].append(target_ship)
                        df_as["N"].append(ais_cur1-curN)
                        df_as["E"].append(ais_cur2-curE)
                        df_as_grid[0][grid0][grid1] += np.abs(ais_cur1-curN)
                        df_as_grid[1][grid0][grid1] += np.abs(ais_cur2-curE)
                        count_grid[grid0][grid1] += 1 
                        print(f'ais: N={ais_cur1}, E={ais_cur2}')
                        print(f'ship: N={curN}, E={curE}')

                    # ログの出力
                    if len(df_as)>0:
                        N = np.array(df_as["N"])
                        E = np.array(df_as["E"])
                        n_df_data = len(N)
                        print(f'as: N={np.mean(np.abs(N))}, E={np.mean(np.abs(E))}, ndata={n_df_data}')

    # データの整理
    dtidxs = np.array(df_as["Dtidx"])
    shipName = np.array(df_as["ShipName"])
    N = np.array(df_as["N"])
    E = np.array(df_as["E"])
    tf = count_grid>0
    df_as_grid[0][tf] /= count_grid[tf]
    df_as_grid[1][tf] /= count_grid[tf]

    # 性能の結果
    print(f'as: N={np.mean(np.abs(N))}, E={np.mean(np.abs(E))}, ndata={len(df_as)}')

    # 時間の違いの性能の結果
    plt.figure()
    plt.scatter(dtidxs, N, label="North")
    plt.scatter(dtidxs, E, label="East")
    plt.legend()
    savepath = "./logs/analysis_diff-ais-ship_dtidx.png"
    plt.savefig(savepath)

    # 船の違いの性能の結果
    plt.figure(figsize=(16, 16))
    N2 = []
    E2 = []
    for target_ship in target_ships:
        tf = shipName == target_ship
        N2.append(np.mean(np.abs(N[tf])))
        E2.append(np.mean(np.abs(E[tf])))
    x = np.arange(len(target_ships))
    width = 0.4
    plt.bar(x-width/2, N2, color='b', label="North", tick_label=target_ships)
    plt.bar(x+width/2, E2, color='r', label="East", tick_label=target_ships)
    plt.legend()
    savepath = "./logs/analysis_diff-ais-ship_shipname.png"
    plt.savefig(savepath)

    # 船ごとの使用したデータ数
    plt.figure(figsize=(16, 16))
    counts = []
    for target_ship in target_ships:
        tf = shipName == target_ship
        counts.append(np.sum(tf))
    plt.bar(target_ships, counts, color='b')
    savepath = "./logs/analysis_shipDataNum.png"
    plt.savefig(savepath)

    # 座標の違いの性能の結果
    plt.figure()
    fig, axes = plt.subplots(1, 3)
    im = axes[0].imshow(df_as_grid[0])
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(df_as_grid[1])
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(count_grid)
    fig.colorbar(im, ax=axes[2])

    axes[0].set_title("North")
    axes[1].set_title("East")
    axes[2].set_title("Num of Data")
    savepath = "./logs/analysis_diff-ais-ship_grid.png"
    plt.savefig(savepath)


if __name__ == '__main__':
    analysis()
    