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

from entire_utils import * 
from kf_params import *

dt_year = year = 2015
dt_month = month = 9
n_day = 4 #nday_month(month) - 1
n_hour = 12 
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True
log_path = path_save + f"{0.1}"

done_ship = []
is_light = True
s = ''

def plot_mappedData(al):
    #for s in range(0, 1):
    day = n_day

    hour = n_hour - 1     
    dt = datetime.datetime(year, month, day, hour, 0, 0)
    dtidx = date_to_dtidx(base_dt, dt)
    
    data  = al.load_cur(dtidx)
    an_map, ae_map = [data[key] for key in ais_keys]

    an_vec = kurosio_filter_pooled(an_map, nan_map_pooled, is_pooled=True)
    ae_vec = kurosio_filter_pooled(ae_map, nan_map_pooled, is_pooled=True)
    print(f'not nan nmap: {np.sum(an_map==an_map)}')
    print(f'not nan emap: {np.sum(ae_map==ae_map)}')
    print(f'not nan n: {np.sum(an_vec==an_vec)}')
    print(f'not nan e: {np.sum(ae_vec==ae_vec)}')
    print(f'n shape: {an_vec.shape}')
    print(f'e shape: {ae_vec.shape}')

    #an_map = an_map * nan_map_pooled
    #ae_map = ae_map * nan_map_pooled
    an_map = kurosio_vec_to_map_pooled(an_vec, nan_map_pooled) * nan_map_pooled
    ae_map = kurosio_vec_to_map_pooled(ae_vec, nan_map_pooled) * nan_map_pooled

    print(f'ais n {an_map.shape}')
    print(f'ais e {ae_map.shape}')


    print(f'ais n {np.nanmax(an_map)}')
    print(f'ais e {np.nanmax(ae_map)}')

    print(f'Min value')
    print(f'ais n {np.nanmin(an_map)}')
    print(f'ais e {np.nanmin(ae_map)}')


    df_ae_map = pd.DataFrame(an_map)
    df_an_map = pd.DataFrame(ae_map)
    df_nan_map = pd.DataFrame(nan_map_pooled)
    
    path = osp.join(log_path, f'ais_n{year}{month:02}{day:02}{hour:02}.csv')
    df_an_map.to_csv(path, index=False, header=False)
    # np.savetxt(path, kalman_n_map, delimiter=',', fmt='%f')
    
    df_ae_map.to_csv(path, index=False, header=False)
    # np.savetxt(path, kalman_e_map, delimiter=',', fmt='%f')

    df_nan_map.to_csv(path, index=False, header=False)

    max_value = 3 
    min_value = -3

    data = np.concatenate([an_map])
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

    data = np.concatenate([ae_map])
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

    print(f"Finished saving mapped date.")



if __name__ == '__main__':

    ais_keys = ['n', 'e']
    al = atc.AISLoader(year, month, osp.join(path_ais, 'ais_files'), pkl_path=path_ais)
    al.set_keys(ais_keys)
    plot_mappedData(al)
    