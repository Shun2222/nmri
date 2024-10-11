import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
import seaborn as sns
from Ais4ToCurForKalmanTime import ais4
import Ais4ToCurForKalmanTime as atc
from experiment_utils import * 
sns.set(font_scale=4)

year = 2015
month = 9
n_day = 15 #nday_month(month) - 1
n_hour = 12 
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True

print(f'Loading data')

kl = KalmanLogLoader(2015, 9, 20)
log_path = r"E:\shunsukeE\result\kalman-pooled3-Q0.01"
path_ais = r'E:\shunsukeE\data\ais\1509-ais4s-pkls-pooled3'
kalman_keys = ['X', 'JCOPE', 'Z']
kl.set_path(log_path)

ais_keys = ['n', 'e']
al = atc.AISLoader(year, month, osp.join(path_ais, 'ais_files'), path_ais)
al.set_keys(ais_keys)

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
    print(jn_map.shape)
    print(an_map.shape)

    lat11 = 180#int(map_pooled_size[0]-kurosio_latidx_range1[0]/2)
    lat00 = 252#int(map_pooled_size[0]-kurosio_latidx_range2[1]/2)
    lon1 = 136#int(map_pooled_size[1]-kurosio_lonidx_range[0]/2)
    lon0 = 255#int(map_pooled_size[1]-kurosio_lonidx_range[1]/2)
    dif = 3 

    n_map = n_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    e_map = e_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    jn_map = jn_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    je_map = je_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    kjn_map = kjn_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    kje_map = kje_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    an_map = an_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
    ae_map = ae_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T

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
