import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os 
import os.path as osp
import pandas as pd
import seaborn as sns
import datetime 
import tqdm

from Ais4ToCur import ais4
import Ais4ToCur as atc

from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from entire_kf_params import *
from entire_utils import *


dt_year = year = 2015
dt_month = month = 9
n_day = 15 #nday_month(month) - 1
n_hour = 12 
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True

done_ship = []
is_light = True
s = 'v2' #v2 or 0

path_kalman = log_path = r"E:\shunsukeE\result\kalman-entire-pooled6-Q0.1"
path_ship = r"E:\shunsukeE\data\shiplog/"
path_ais = r'E:\shunsukeE\data\ais\1509-ais4s-pkls-pooled6-entire'
path_jcope = fr'E:\shunsukeE\data\eas2'

def plot_mappedData(kl, al):
    #for s in range(0, 1):
    for s in ['v1', 'v2']:
        kalman_n = []
        kalman_e = []
        day = n_day
        print(f's:{s}, day:{day}')
        # TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']
        if is_light:
            TF = np.array([True] * 3789)
        else:
            TF = np.array([True] * 9808)

        start_hour = 0 if day == 1 else 1
        for hour in range(start_hour, n_hour):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue

        hour = n_hour - 1     
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
        else:
            n_map = kurosio_vec_to_map(kalman_n, nan_map) * nan_map
            e_map = kurosio_vec_to_map(kalman_e, nan_map) * nan_map
            jn_map = kurosio_vec_to_map(jcope_n, nan_map) * nan_map
            je_map = kurosio_vec_to_map(jcope_e, nan_map) * nan_map
            
        
        data  = al.load_cur(dtidx)
        an_map, ae_map = [data[key] for key in ais_keys]
        print(f'ais n {an_map.shape}')
        print(f'ais e {ae_map.shape}')

        lat00 = int(map_pooled_size[0]-kurosio_latidx_range1[0]/2)
        lat11 = int(map_pooled_size[0]-kurosio_latidx_range2[1]/2)
        lon0 = int(map_pooled_size[1]-kurosio_lonidx_range[0]/2)
        lon1 = int(map_pooled_size[1]-kurosio_lonidx_range[1]/2)
        dif = 3 

        # n_map = n_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # e_map = e_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # jn_map = jn_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # je_map = je_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # kjn_map = kjn_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # kje_map = kje_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # an_map = an_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
        # ae_map = ae_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T


        print(f'Max value')
        print(f'n {np.nanmax(n_map)}')
        print(f'e {np.nanmax(e_map)}')
        print(f'jcope n {np.nanmax(jn_map)}')
        print(f'jcope e {np.nanmax(je_map)}')
        print(f'ais n {np.nanmax(an_map)}')
        print(f'ais e {np.nanmax(ae_map)}')

        print(f'Min value')
        print(f'n {np.nanmin(n_map)}')
        print(f'e {np.nanmin(e_map)}')
        print(f'jcope n {np.nanmin(jn_map)}')
        print(f'jcope e {np.nanmin(je_map)}')
        print(f'ais n {np.nanmin(an_map)}')
        print(f'ais e {np.nanmin(ae_map)}')


        df_n_map = pd.DataFrame(n_map)
        df_e_map = pd.DataFrame(e_map)    
        df_jn_map = pd.DataFrame(jn_map)
        df_je_map = pd.DataFrame(je_map)
        df_kjn_map = pd.DataFrame(kjn_map)
        df_kje_map = pd.DataFrame(kje_map)
        df_ae_map = pd.DataFrame(an_map)
        df_an_map = pd.DataFrame(ae_map)
        df_nan_map = pd.DataFrame(nan_map_pooled)
        
        path = osp.join(log_path, f'kalman_n{year}{month:02}{day:02}{hour:02}.csv')
        df_n_map.to_csv(path, index=False, header=False)
        path = osp.join(log_path, f'jcope_n{year}{month:02}{day:02}{hour:02}.csv')
        df_jn_map.to_csv(path, index=False, header=False)
        path = osp.join(log_path, f'diff_kalman_jcope_n{year}{month:02}{day:02}{hour:02}.csv')
        df_kjn_map.to_csv(path, index=False, header=False)
        path = osp.join(log_path, f'ais_n{year}{month:02}{day:02}{hour:02}.csv')
        df_an_map.to_csv(path, index=False, header=False)
        # np.savetxt(path, kalman_n_map, delimiter=',', fmt='%f')
        
        path = osp.join(log_path, f'kalman_e{year}{month:02}{day:02}{hour:02}.csv')
        df_e_map.to_csv(path, index=False, header=False)
        path = osp.join(log_path, f'jcope_e{year}{month:02}{day:02}{hour:02}.csv')
        df_je_map.to_csv(path, index=False, header=False)
        path = osp.join(log_path, f'diff_kalman_jcope_e{year}{month:02}{day:02}{hour:02}.csv')
        df_kje_map.to_csv(path, index=False, header=False)
        path = osp.join(log_path, f'ais_e{year}{month:02}{day:02}{hour:02}.csv')
        df_ae_map.to_csv(path, index=False, header=False)
        # np.savetxt(path, kalman_e_map, delimiter=',', fmt='%f')


        path = osp.join(log_path, f'nan_map.csv')
        df_nan_map.to_csv(path, index=False, header=False)

        max_value = 3 
        min_value = -3

        data = np.concatenate([an_map, jn_map])
        data = np.concatenate([data, n_map])
        plt.figure(figsize=(24, 24))
        plt.title(f'N {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'N_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(log_path, path)
        plt.savefig(path)
        plt.close()

        data[data<min_value] = np.nan
        data[data>max_value] = np.nan
        plt.figure(figsize=(24, 24))
        plt.title(f'N Filtered by {min_value}<=data<={max_value} {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'N_Filtered_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(log_path, path)
        plt.savefig(path)
        plt.close()

        data = np.concatenate([ae_map, je_map])
        data = np.concatenate([data, e_map])
        plt.figure(figsize=(24, 24))
        plt.title(f'E {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'E_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(log_path, path)
        plt.savefig(path)
        plt.close()

        data[data<min_value] = np.nan
        data[data>max_value] = np.nan
        plt.figure(figsize=(24, 24))
        plt.title(f'E Filtered by {min_value}<=data<={max_value} {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'E_Filtered_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(log_path, path)
        plt.savefig(path)
        plt.close()

        data = np.concatenate([kjn_map, kje_map])
        plt.figure(figsize=(24, 24))
        plt.title(f'NE {year}{month:02}{day:02} (AIS, JCOPE, KALMAN)')
        sns.heatmap(data, cbar=True)
        path = f'kalman_jcope_NE_{year}{month:02}{day:02}-{s}.png'
        path = osp.join(log_path, path)
        plt.savefig(path)
        plt.close()
    print(f"Finished saving mapped date.")


def diff_results(kl):
    data = kl.load_kalmanLog_day(1, s, keys=['X'])
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

    max_day = nday_month(month) - 1
    for day in range(1, max_day+1):
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

            data = kl.load_kalmanLog_day(day, s, keys=['X', 'JCOPE'])
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
                    if day==1:
                        print(f'There is no kalman data (dtidx = {dtidx}).')
                        continue
                    data = kl.load_kalmanLog_day(day-1, s, keys=['X', 'JCOPE'])
                    if not dtidx in data['X'].keys():
                        print(f'There is no kalman data (dtidx = {dtidx}).')
                        continue

                kfidx = get_kfidx(TFs[0], idx)
                kalman_cur1 = data['X'][dtidx][kfidx]
                kalman_cur2 = data['X'][dtidx][kfidx+n_data]

                jcope_cur1 = data['JCOPE'][dtidx][kfidx]
                jcope_cur2 = data['JCOPE'][dtidx][kfidx+n_data]

                hour = dtidx_to_date(base_dt, int(dtidx)).hour
                path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}X.csv"
                dataN = pd.read_csv(path, encoding="cp932", header=None)
                ais_cur1 = dataN.values[grid0][grid1]
                
                path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}Y.csv"
                dataE = pd.read_csv(path, encoding="cp932", header=None)
                ais_cur2 = dataE.values[grid0][grid1]
                
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
                

    print(f'aj: {np.mean(np.abs(df_aj))}')
    print(f'ak: {np.mean(np.abs(df_ak))}')
    print(f'as: {np.mean(np.abs(df_as))}')
    print(f'kj: {np.mean(np.abs(df_kj))}')
    print(f'sj: {np.mean(np.abs(df_sj))}')
    print(f'ks: {np.mean(np.abs(df_ks))}')
    for target_ship in target_ships:
        a = np.mean(diff_ship[target_ship][0])
        b = np.mean(diff_ship[target_ship][1])
        c = diff_ship[target_ship][2]
        print(f'{target_ship}: {round(np.mean(a), 2)}, {round(np.mean(b), 2)}, {c}')


if __name__ == '__main__':
    kl = KalmanLogLoader(2015, 9, 20)
    kalman_keys = ['X', 'JCOPE', 'Z', 'Target']
    kl.set_path(log_path)

    ais_keys = ['n', 'e']
    al = atc.AISLoader(year, month, osp.join(path_ais, 'ais_files'), pkl_path=path_ais)
    al.set_keys(ais_keys)
    plot_mappedData(kl, al)
    diff_results(kl)
    