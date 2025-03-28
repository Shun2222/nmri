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
from KalmanFilterProgram.utils.kalman_funcs import *
from scipy import stats


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
    ais_keys = ['N', 'E', 'lambda1', 'lambda2']


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

def eval_kalmanLog(datas, use_lambda=[0, 0], plot=True):
    from sklearn.metrics.pairwise import cosine_similarity

    lambda_data = np.array(datas["lambda"])
    lambda1 = lambda_data.T[0]
    lambda2 = lambda_data.T[1]
    tf_lambda1 = lambda1 > use_lambda[0]
    tf_lambda2 = lambda2 > use_lambda[1]
    tf_lambda = tf_lambda1 & tf_lambda2
    lambda1 = lambda1[tf_lambda]
    lambda2 = lambda2[tf_lambda]


    print('\n')
    print(f"NumData:{np.sum(tf_lambda)}")
    print('--------------------SHIP-KALMAN ANALYSIS--------------------\n')
    ship = np.array(datas['ship']).T
    ship = np.array([ship[0][tf_lambda], ship[1][tf_lambda]])
    kalman = np.array(datas['kalman']).reshape(len(tf_lambda.T), 2).T
    kalman = np.array([kalman[0][tf_lambda], kalman[1][tf_lambda]])
    #cos_simN = cosine_similarity(ship.T[0].reshape(1, len(ship)), kalman.T[0].reshape(1, len(ship)))
    #cos_simE = cosine_similarity(ship.T[1].reshape(1, len(ship)), kalman.T[1].reshape(1, len(ship)))
    #print(f"cosine similarity : (N, E) = ({cos_simN}, {cos_simE})\n")
    res = np.mean(np.abs(ship - kalman), axis=1)
    res2 = np.sqrt(np.mean((ship - kalman)**2, axis=1))
    print(f"mean(|a-b|) : (N, E) = ({np.round(res[0], 4)}, {np.round(res[1], 4)})\n")
    print(f"sqrt(mean((a-b)^2)) : (N, E) = ({np.round(res2[0], 4)}, {np.round(res2[1], 4)})\n")
    mean_kalman = np.nanmean(kalman, axis=1)
    mean_ship = np.nanmean(ship, axis=1)
    r = [np.sum((kalman[i]-mean_kalman[i])*(ship[i]-mean_ship[i]))/np.sqrt((np.sum((kalman[i]-mean_kalman[i])**2))*(np.sum((ship[i]-mean_ship[i])**2))) for i in range(len(kalman))]
    print(f"relationship: {np.round(r[0], 4)}, {np.round(r[1], 4)}")

    plt.close()
    if plot:
        plt.title("lmabda1-(|kalman-ship|)")
        plt.scatter(lambda1, np.abs(ship - kalman)[0])
        plt.show()
        plt.title("Lmabda2-(|Kalman-ship|)")
        plt.scatter(lambda2, np.abs(ship - kalman)[1])
        plt.show()

    print('--------------------SHIP-AIS ANALYSIS--------------------\n')
    ais = []
    ais_tf = []
    for ais_data in datas['ais']:
        if ais_data==[] or ais_data[1]!=ais_data[1] or ais_data[0]!=ais_data[0]:
            ais_tf.append(False)
            continue
        else:
            ais_tf.append(True)
        ais.append(ais_data)
    ais = np.array(ais)
    if len(ais) != 0:
        #ship2 = ship[ais_tf]
        ship2 = ship
        ais = np.array(datas['ais']).reshape(len(tf_lambda.T), 2).T
        ais = np.array([ais[0][tf_lambda], ais[1][tf_lambda]])
        #cos_simN = cosine_similarity(ship2.T[0].reshape(1, len(ais)), ais.T[0].reshape(1, len(ais)))
        #cos_simE = cosine_similarity(ship2.T[1].reshape(1, len(ais)), ais.T[1].reshape(1, len(ais)))
        #print(f"cosine similarity : (N, E) = ({cos_simN}, {cos_simE})")
        res = np.nanmean(np.abs(ship2 - ais), axis=1)
        res2 = np.sqrt(np.mean((ship2 - ais)**2, axis=1))
        ais_res = res
        print(f"mean(|a-b|) : (N, E) = ({np.round(res[0], 4)}, {np.round(res[1], 4)})")
        print(f"sqrt(mean((a-b)^2)) : (N, E) = ({np.round(res2[0], 4)}, {np.round(res2[1], 4)})\n")
        mean_ais = np.nanmean(ais, axis=1)
        mean_ship = np.nanmean(ship, axis=1)
        r = [np.sum((ais[i]-mean_ais[i])*(ship[i]-mean_ship[i]))/np.sqrt((np.sum((ais[i]-mean_ais[i])**2))*(np.sum((ship[i]-mean_ship[i])**2))) for i in range(len(ais))]
        print(f"relationship: {np.round(r[0], 4)}, {np.round(r[1], 4)}")
        tf = np.abs(ais)>4
        tf = np.array([any(tf2) for tf2 in tf])
        day = np.array(datas['day'])
        grid = np.array(datas['grid'])
        #print(day[np.array(ais_tf)][tf].flatten())
        #print(grid[np.array(ais_tf)][tf])
    else:
        print(f"AIS is not exist!\n")
    
    #abs_ais = np.abs(ship - ais)
    #print(np.max(lambda2[abs_ais[1] > 1]))
    if plot:
        plt.title("Lmabda1-(|ais-ship|)")
        plt.scatter(lambda1, np.abs(ship - ais)[0])
        plt.show()
        plt.title("Lmabda2-(|ais-ship|)")
        plt.scatter(lambda2, np.abs(ship - ais)[1])
        plt.show()

    if use_ais_removed_bad_mmsi:
        print('--------------------SHIP-Default AIS ANALYSIS--------------------\n')
        ais2_tf = []
        for ais2_data in datas['ais2']:
            if ais2_data==[] or ais2_data[1]!=ais2_data[1] or ais2_data[0]!=ais2_data[0]:
                ais2_tf.append(False)
                continue
            else:
                ais2_tf.append(True)
        ais2_tf = np.array(ais2_tf)
        ais2 = np.array(datas['ais2']).T
        ais2 = np.array([ais2[0][ais2_tf & tf_lambda], ais2[1][ais2_tf & tf_lambda]])
        if len(ais2) != 0:
            ship2 = np.array(datas['ship']).T
            ship2 = np.array([ship2[0][ais2_tf & tf_lambda], ship2[1][ais2_tf & tf_lambda]])
            #cos_simN = cosine_similarity(ship2[0].reshape(1, len(ais2)), ais2[0].reshape(1, len(ais2)))
            #cos_simE = cosine_similarity(ship2[1].reshape(1, len(ais2)), ais2[1].reshape(1, len(ais2)))
            #print(f"cosine similarity : (N, E) = ({cos_simN}, {cos_simE})")
            res = np.mean(np.abs(ship2 - ais2), axis=1)
            res2 = np.sqrt(np.mean((ship2 - ais2)**2, axis=1))
            ais2_res = res
            print(f"mean(|a-b|) : (N, E) = ({np.round(res[0], 4)}, {np.round(res[1], 4)})")
            print(f"sqrt(mean((a-b)^2)) : (N, E) = ({np.round(res2[0], 4)}, {np.round(res2[1], 4)})\n")
        else:
            print(f"AIS2 (default AIS) is not exist!\n")

        mean_ais2 = np.nanmean(ais2, axis=1)
        mean_ship = np.nanmean(ship, axis=1)
        r = [np.sum((ais2[i]-mean_ais2[i])*(ship[i]-mean_ship[i]))/np.sqrt((np.sum((ais2[i]-mean_ais2[i])**2))*(np.sum((ship[i]-mean_ship[i])**2))) for i in range(len(ais2))]
        print(f"relationship: {np.round(r[0], 4)}, {np.round(r[1], 4)}")

        if plot:
            plt.title("Lmabda1-(|original_ais-ship|)")
            plt.scatter(lambda1, np.abs(ship - ais2)[0])
            plt.show()
            plt.title("Lmabda2-(|original_ais-ship|)")
            plt.scatter(lambda2, np.abs(ship - ais2)[1])
            plt.show()

    print('--------------------SHIP-JCOPE ANALYSIS--------------------\n')
    jcope = np.array(datas['jcope']).reshape(len(tf_lambda.T), 2).T
    jcope = np.array([jcope[0][tf_lambda], jcope[1][tf_lambda]])
    #cos_simN = cosine_similarity(ship.T[0].reshape(1, len(jcope)), jcope.T[0].reshape(1, len(jcope)))
    #cos_simE = cosine_similarity(ship.T[1].reshape(1, len(jcope)), jcope.T[1].reshape(1, len(jcope)))
    #print(f"cosine similarity : (N, E) = ({cos_simN}, {cos_simE})\n")
    res = np.mean(np.abs(ship - jcope), axis=1)
    res2 = np.sqrt(np.mean((ship - jcope)**2, axis=1))
    print(f"mean(|a-b|) : (N, E) = ({np.round(res[0], 4)}, {np.round(res[1], 4)})\n")
    print(f"sqrt(mean((a-b)^2)) : (N, E) = ({np.round(res2[0], 4)}, {np.round(res2[1], 4)})\n")
    print('-----------------------------------------------------------\n')

    if plot:
        plt.title("Lmabda1-(|jcope-ship|)")
        plt.scatter(lambda1, np.abs(ship - jcope)[0])
        plt.show()
        plt.title("Lmabda2-(|jcope-ship|)")
        plt.scatter(lambda2, np.abs(ship - jcope)[1])
        plt.show()

    #t_stat, p_value = stats.ttest_rel(ais[0], ais2[0])
    #print(f"t_stat: {t_stat}, p-value:{p_value}")
    #t_stat, p_value = stats.ttest_rel(ais[1], ais2[1])
    #print(f"t_stat: {t_stat}, p-value:{p_value}")

    plt.scatter(ship[0], ais[0])
    plt.show()
    plt.scatter(ship[1], ais[1])
    plt.show()
    #plt.scatter(ship[0], ais2[0])
    #plt.show()
    #plt.scatter(ship[1], ais2[1])
    #plt.show()
    
    abs_kalman = np.abs(ship - kalman)
    abs_jcope = np.abs(ship - jcope)
    print(np.max(lambda2[abs_kalman[1] <= abs_jcope[1]]))
    print(np.min(lambda2[abs_kalman[1] <= abs_jcope[1]]))
    plt.scatter(lambda2[abs_kalman[1] > abs_jcope[1]], abs_kalman[1][abs_kalman[1] > abs_jcope[1]], color='r', alpha=0.3, label="Loss")
    plt.scatter(lambda2[abs_kalman[1] <= abs_jcope[1]], abs_kalman[1][abs_kalman[1] <= abs_jcope[1]], alpha=0.3, label="Win")
    plt.legend()
    if plot:
        plt.show()
    else:
        plt.close()


    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    colors = {'ship': 'r', 
              'kalman' : 'b', 
              'ais' : 'g', 
              'ais2' : 'gray', 
              'jcope' : 'purple'}
    alpha = 0.8
    for ax in axes:
        if ax==axes[-1]:
            ax.scatter(ship[0], ship[1], color=colors['ship'], alpha=alpha, label='Shiplog')
        else:
            ax.scatter(ship[0], ship[1], color=colors['ship'], alpha=alpha)
    axes[0].scatter(kalman[0], kalman[1], color=colors['kalman'], alpha=alpha)
    axes[3].scatter(kalman[0], kalman[1], color=colors['kalman'], alpha=alpha, label='Kalman')
    if use_ais_removed_bad_mmsi:
        if len(ais)!=0:
            axes[1].scatter(ais[0], ais[1], color=colors['ais'], alpha=alpha)
            axes[3].scatter(ais[0], ais[1], color=colors['ais'], alpha=alpha, label='Removed Bad MMSI AIS')
        if len(ais2)!=0:
            axes[1].scatter(ais2[0], ais2[1], color=colors['ais2'], alpha=alpha)
            axes[3].scatter(ais2[0], ais2[1], color=colors['ais2'], alpha=alpha, label='Default AIS')
    else:
        if len(ais)!=0:
            axes[1].scatter(ais[0], ais[1], color=colors['ais'], alpha=alpha)
            axes[3].scatter(ais[0], ais[1], color=colors['ais'], alpha=alpha, label='AIS')
    axes[2].scatter(jcope[0], jcope[1], color=colors['jcope'], alpha=alpha)
    axes[3].scatter(jcope[0], jcope[1], color=colors['jcope'], alpha=alpha, label='JCOPE')

    for ax in axes:
        ax.set_xlabel('N')
        ax.set_ylabel('E')
    axes[0].set_title('Ploted Kalman results and Shiplog')
    axes[1].set_title('Ploted AIS and Shiplog')
    axes[2].set_title('Ploted JCOPE and Shiplog')
    axes[3].set_title('Ploted Kalman results, AIS, JCOPE and Shiplog')
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.close()

def make_lambdacur_pkl(kl, al, al2, jl):
    data = kl.load_kalmanLog_day(1, s, keys=['X', 'K', 'H'])

    files = os.listdir(path_ship)
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))]

    df_aj = []
    df_as = []
    df_ak = []
    df_a2j = []
    df_a2s = []
    df_a2k = []
    df_sj = []
    df_kj = []
    df_ks = []
    dtidxs = []
    grids = []
    diff_ship = {}

    keys = ['day', 'grid', 'ship', 'kalman', 'ais', 'ais2','jcope', 'lambda','original_lambda']
    all_data = {}
    for key in keys:
        all_data[key] = []

    for target_ship in target_ships:
        diff_ship[target_ship] = [[], [], 0]


    colors = {'ship': 'r', 
              'kalman' : 'b', 
              'ais' : 'g', 
              'ais2' : 'gray', 
              'jcope' : 'purple'}

    good_cur1_count = np.zeros((nan_map_pooled.shape))*nan_map_pooled
    bad_cur1_count = np.zeros((nan_map_pooled.shape))*nan_map_pooled
    good_cur2_count = np.zeros((nan_map_pooled.shape))*nan_map_pooled
    bad_cur2_count = np.zeros((nan_map_pooled.shape))*nan_map_pooled
    compair_cur1_ks = np.zeros((nan_map_pooled.shape))*nan_map_pooled
    compair_cur2_ks = np.zeros((nan_map_pooled.shape))*nan_map_pooled

    is_first = [True, True, True]
    n_data = -1
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
            #path_log = osp.join(path_ship, target_ship, '2015')
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

                data = kl.load_kalmanLog_day(day, s, keys=['X', 'JCOPE', 'K', 'H'])
                if n_data == -1:
                    n_data = int((data['X'][1].shape[0] - 1)/2)
                    TFs = [np.array([True]*n_data)]

                #if not dtidx in data['X'].keys() and day!=1:
                if not dtidx in data['X'].keys():
                    if day!=1:
                        data = kl.load_kalmanLog_day(day-1, s, keys=['X', 'JCOPE', 'K', 'H'])
                    if not dtidx in data['X'].keys():
                        print(f'There is no kalman data (dtidx = {dtidx}).')
                        continue

                kfidx = get_kfidx(TFs[0], idx)
                kalman_cur1 = data['X'][dtidx][kfidx]
                kalman_cur2 = data['X'][dtidx][kfidx+n_data]

                jcope_cur1 = data['JCOPE'][dtidx][kfidx]
                jcope_cur2 = data['JCOPE'][dtidx][kfidx+n_data]

                dtidx2 = dtidx
                try:
                    ais_data = al.load_ais_dtidx(dtidx2, keys=["Cur1", "Cur2", "Lambda1", "Lambda2", "Phi1", "Phi2"])
                except:
                    continue
                if use_ais_removed_bad_mmsi:
                    ais_data2 = al2.load_ais_dtidx(dtidx2, keys=["cur1", "cur2", "lambda1", "lambda2", "psi1", "psi2"])
                    if "cur1" in ais_data2:
                        ais_data3 = {}
                        key2 = ["Cur1", "Cur2", "Lambda1", "Lambda2", "Phi1", "Phi2"]
                        for keyi, key in enumerate(["cur1", "cur2", "lambda1", "lambda2", "psi1", "psi2"]):
                            ais_data3[key2[keyi]] = ais_data2[key]
                        ais_data2 = ais_data3
                #print(f"dtidx: {dtidx2}")
                if "cur1" in ais_data:
                    ais_data3 = {}
                    key2 = ["Cur1", "Cur2", "Lambda1", "Lambda2", "Phi1", "Phi2"]
                    for keyi, key in enumerate(["cur1", "cur2", "lambda1", "lambda2", "psi1", "psi2"]):
                        ais_data3[key2[keyi]] = ais_data[key]
                    ais_data = ais_data3
                if type(ais_data["Cur1"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    #print(f"N shape: {dataN.shape}")
                    ais_cur1 = ais_data["Cur1"][grid0][grid1]
                    
                    ais_cur2 = ais_data["Cur2"][grid0][grid1]

                    lambda1 = ais_data["Lambda1"]
                    lambda1 = lambda1[grid0][grid1]
                    lambda2 = ais_data["Lambda2"]
                    lambda2 = lambda2[grid0][grid1]

                    if use_ais_removed_bad_mmsi:
                        ais_cur1_2 = ais_data2["Cur1"][grid0][grid1]
                        ais_cur2_2 = ais_data2["Cur2"][grid0][grid1]
                        lambda1_2 = ais_data2["Lambda1"]
                        lambda2_2 = ais_data2["Lambda2"]
                    
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                    #elif not (lambda2 > 100 and lambda1 > 10):
                    elif not (lambda2 > 100):
                        print(f"lambda2 is low {lambda2}, cur: {ais_cur1}, {ais_cur2}")

                    alpha = 0.8
                    #available_ais = lambda2 > 100  and ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    #available_ais = lambda1 > 5 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    #available_ais = lambda2 > 100 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    #available_ais = lambda2 > 100 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    available_ais = ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    #available_ais = ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    if available_ais:
                        cur_ship1 = curN*np.cos(ais_data["Phi1"][grid0][grid1])+curE*np.sin(ais_data["Phi1"][grid0][grid1])
                        cur_ship2 = curN*np.cos(ais_data["Phi2"][grid0][grid1])+curE*np.sin(ais_data["Phi2"][grid0][grid1])
                        jcope_cur1_2 = jcope_cur1*np.cos(ais_data["Phi1"][grid0][grid1])+jcope_cur2*np.sin(ais_data["Phi1"][grid0][grid1])
                        jcope_cur2 = jcope_cur1*np.cos(ais_data["Phi2"][grid0][grid1])+jcope_cur2*np.sin(ais_data["Phi2"][grid0][grid1])
                        jcope_cur1 = jcope_cur1_2
                        kalman_curN = kalman_cur1
                        kalman_curE = kalman_cur2
                        kalman_cur1_2 = kalman_cur1*np.cos(ais_data["Phi1"][grid0][grid1])+kalman_cur2*np.sin(ais_data["Phi1"][grid0][grid1])
                        kalman_cur2 = kalman_cur1*np.cos(ais_data["Phi2"][grid0][grid1])+kalman_cur2*np.sin(ais_data["Phi2"][grid0][grid1])
                        kalman_cur1 = kalman_cur1_2

                        df_aj.append([ais_cur1-jcope_cur1, ais_cur2-jcope_cur2])
                        df_as.append([ais_cur1-cur_ship1, ais_cur2-cur_ship2])
                        df_ak.append([ais_cur1-kalman_cur1, ais_cur2-kalman_cur2])

                    if use_ais_removed_bad_mmsi:
                        if ais_cur1_2==ais_cur1_2 and ais_cur2_2==ais_cur2_2:
                            df_a2j.append([ais_cur1_2-jcope_cur1, ais_cur2_2-jcope_cur2])
                            df_a2s.append([ais_cur1_2-curN, ais_cur2_2-curE])
                            df_a2k.append([ais_cur1_2-kalman_cur1, ais_cur2_2-kalman_cur2])

                            #axes[1].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha)
                            if is_first[2]:
                                #axes[3].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha, label='AIS')
                                is_first[2] = False
                            else:
                                #axes[3].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha)
                                None
                    if available_ais:
                        df_sj.append([jcope_cur1-cur_ship1, jcope_cur2-cur_ship2])    
                        df_kj.append([jcope_cur1-kalman_cur1, jcope_cur2-kalman_cur2])    
                        df_ks.append([kalman_cur1-cur_ship1, kalman_cur2-cur_ship2]) 
                        diffs.append([kalman_cur1-cur_ship1, kalman_cur2-cur_ship2])
                        if np.abs(df_ks[-1][0]) > np.abs(df_sj[-1][0]):
                            bad_cur1_count[grid0][grid1] += 1
                        else:
                            good_cur1_count[grid0][grid1] += 1
                        if np.abs(df_ks[-1][1]) > np.abs(df_sj[-1][1]):
                            bad_cur2_count[grid0][grid1] += 1
                        else:
                            good_cur2_count[grid0][grid1] += 1
                        compair_cur1_ks[grid0][grid1] += np.abs(df_sj[-1][0]) - np.abs(df_ks[-1][0])
                        compair_cur2_ks[grid0][grid1] += np.abs(df_sj[-1][1]) - np.abs(df_ks[-1][1])

                        all_data['day'].append([day])
                        all_data['grid'].append([grid0, grid1])
                        all_data['ship'].append([cur_ship1, cur_ship2])
                        all_data['kalman'].append([kalman_cur1, kalman_cur2])
                        all_data['ais'].append([ais_cur1, ais_cur2])
                        all_data['lambda'].append([lambda1, lambda2])
                        if use_ais_removed_bad_mmsi:
                            all_data['ais2'].append([ais_cur1_2, ais_cur2_2])
                            all_data['original_lambda'].append([lambda1_2, lambda2_2])
                        all_data['jcope'].append([jcope_cur1, jcope_cur2])

                        is_first[0] = False
                        available_count += 1 

            if available_count>0:
                diffs = np.array(diffs)
                diffs = diffs.reshape(diffs.shape[0], diffs.shape[1])
                diff_N = diffs.T[0]
                diff_E = diffs.T[1]
                diff_ship[target_ship][0] += diff_N.tolist()
                diff_ship[target_ship][1] += diff_E.tolist()
                diff_ship[target_ship][2] += available_count
    pklfilename = osp.join(path_log, 'analysis_data.pkl')
    pickle_dump(all_data, pklfilename)
    print(f'Saved in {pklfilename}')
                
    vmax = np.nanmax([bad_cur1_count, bad_cur2_count, good_cur1_count, good_cur2_count])
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    axes[0].set_title("Good pos cur1 count")
    im = axes[0].imshow(good_cur1_count, vmin=0, vmax=vmax)
    axes[1].set_title("Good pos cur2 count")
    im = axes[1].imshow(good_cur2_count, vmin=0, vmax=vmax)
    axes[2].set_title("Bad pos cur1 count")
    im = axes[2].imshow(bad_cur1_count, vmin=0, vmax=vmax)
    axes[3].set_title("Bad pos cur2 count")
    im = axes[3].imshow(bad_cur2_count, vmin=0, vmax=vmax)
    fig.colorbar(im, axes[4])
    plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    vmax = np.nanmax([np.abs(np.nanmin(compair_cur1_ks)), np.abs(np.nanmax(compair_cur1_ks))])
    axes[0].set_title("Cur1 (higher value is higher score)")
    im = axes[0].imshow(compair_cur1_ks, cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, axes[1])

    vmax = np.nanmax([np.abs(np.nanmin(compair_cur2_ks)), np.abs(np.nanmax(compair_cur2_ks))])
    axes[2].set_title("Cur2 (higher value is higher score)")
    im = axes[2].imshow(compair_cur2_ks, cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, axes[3])
    plt.show()


    print(f'aj: {np.mean(np.abs(df_aj))}, ndata={len(df_aj)}')
    print(f'ak: {np.mean(np.abs(df_ak))}, ndata={len(df_ak)}')
    print(f'as: {np.mean(np.abs(df_as))}, ndata={len(df_as)}')
    print(f'a2j: {np.mean(np.abs(df_a2j))}, ndata={len(df_a2j)}')
    print(f'a2k: {np.mean(np.abs(df_a2k))}, ndata={len(df_a2k)}')
    print(f'a2s: {np.mean(np.abs(df_a2s))}, ndata={len(df_a2s)}')
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
           [len(df_aj), len(df_ak),len(df_as),len(df_kj),len(df_sj),len(df_ks)],\
            all_data

def make_analysis_lambdacur_pkl(kl, al, al2, jl):
    data = kl.load_kalmanLog_day(1, s, keys=['X', 'K', 'H'])

    files = os.listdir(path_ship)
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))]

    df_aj = []
    df_as = []
    df_ak = []
    df_a2j = []
    df_a2s = []
    df_a2k = []
    df_sj = []
    df_kj = []
    df_ks = []
    dtidxs = []
    grids = []
    diff_ship = {}

    keys = ['day', 'grid', 'ship', 'kalman', 'ais', 'ais2','jcope','lambda', 'origin_lambda']
    all_data = {}
    for key in keys:
        all_data[key] = []

    for target_ship in target_ships:
        diff_ship[target_ship] = [[], [], 0]


    colors = {'ship': 'r', 
              'kalman' : 'b', 
              'ais' : 'g', 
              'ais2' : 'gray', 
              'jcope' : 'purple'}

    is_first = [True, True, True]
    n_data = -1
    for day in range(1, n_day+1):
    #for day in range(23, 23+1):
        jcope_n, jcope_e = jl.load_jcope_day(day)
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
            #path_log = osp.join(path_ship, target_ship, '2015')
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

                dtidx_day = dtidx_to_date(dtidx).day
                if dtidx<0 or dtidx_day>MAX_DAY:
                    print(f'Out of dtidx ( dtidx = {dtidx}) ).')
                    continue

                if idx==-1:
                    print(f'Out of idx ( grid = ({grid0},{grid1}) ).')
                    continue

                data = kl.load_kalmanLog_day(day, s, keys=['X', 'JCOPE', 'K', 'H'])
                if n_data == -1:
                    n_data = int((data['X'][1].shape[0] - 1)/2)
                    TFs = [np.array([True]*n_data)]

                #if not dtidx in data['X'].keys() and day!=1:
                if not dtidx in data['X'].keys():
                    if day!=1:
                        data = kl.load_kalmanLog_day(day-1, s, keys=['X', 'JCOPE', 'K', 'H'])
                    if not dtidx in data['X'].keys():
                        print(f'There is no kalman data (dtidx = {dtidx}).')
                        continue

                kfidx = get_kfidx(TFs[0], idx)
                kalman_cur1 = data['X'][dtidx][kfidx]
                kalman_cur2 = data['X'][dtidx][kfidx+n_data]

                jcope_cur1 = data['JCOPE'][dtidx][kfidx]
                jcope_cur2 = data['JCOPE'][dtidx][kfidx+n_data]

                dtidx2 = dtidx
                ais_data = al.load_ais_dtidx(dtidx2, keys=["Cur1", "Cur2", "Lambda1", "Lambda2", "Phi1", "Phi2"])
                if use_ais_removed_bad_mmsi:
                    ais_data2 = al.load_ais_dtidx(dtidx2, keys=["cur1", "cur2", "lambda1", "lambda2", "phi1", "phi2"])
                #print(f"dtidx: {dtidx2}")
                if type(ais_data["Cur1"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    #print(f"N shape: {dataN.shape}")
                    ais_cur1 = ais_data["Cur1"][grid0][grid1]
                    
                    ais_cur2 = ais_data["Cur2"][grid0][grid1]

                    lambda1 = ais_data["Lambda1"]
                    lambda1 = lambda1[grid0][grid1]
                    lambda2 = ais_data["Lambda2"]
                    lambda2 = lambda2[grid0][grid1]

                    if use_ais_removed_bad_mmsi:
                        ais_cur1_2 = ais_data2["cur1"][grid0][grid1]
                        ais_cur2_2 = ais_data2["cur2"][grid0][grid1]
                    
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                    elif not (lambda2 > 100 and lambda1 > 10):
                        print(f"lambda2 is low {lambda2}, cur: {ais_cur1}, {ais_cur2}")

                    alpha = 0.8
                    available_ais = lambda1 > 10 and lambda1 < 10 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2
                    if available_ais:
                    #if ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
                        axes[0].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha, label="AIS")
                        axes[0].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha, label="JCOPE")
                        axes[0].scatter(curN, curE, color='r', label="Ship")
                        axes[0].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha, label="Kalman")

                        cur_ship1 = curN*np.cos(ais_data["Phi1"][grid0][grid1])+curE*np.sin(ais_data["Phi1"][grid0][grid1])
                        cur_ship2 = curN*np.cos(ais_data["Phi2"][grid0][grid1])+curE*np.sin(ais_data["Phi2"][grid0][grid1])
                        jcope_cur1_2 = jcope_cur1*np.cos(ais_data["Phi1"][grid0][grid1])+jcope_cur2*np.sin(ais_data["Phi1"][grid0][grid1])
                        jcope_cur2 = jcope_cur1*np.cos(ais_data["Phi2"][grid0][grid1])+jcope_cur2*np.sin(ais_data["Phi2"][grid0][grid1])
                        jcope_cur1 = jcope_cur1_2
                        kalman_curN = kalman_cur1
                        kalman_curE = kalman_cur2
                        kalman_cur1_2 = kalman_cur1*np.cos(ais_data["Phi1"][grid0][grid1])+kalman_cur2*np.sin(ais_data["Phi1"][grid0][grid1])
                        kalman_cur2 = kalman_cur1*np.cos(ais_data["Phi2"][grid0][grid1])+kalman_cur2*np.sin(ais_data["Phi2"][grid0][grid1])
                        kalman_cur1 = kalman_cur1_2

                        df_aj.append([ais_cur1-jcope_cur1, ais_cur2-jcope_cur2])
                        df_as.append([ais_cur1-cur_ship1, ais_cur2-cur_ship2])
                        df_ak.append([ais_cur1-kalman_cur1, ais_cur2-kalman_cur2])

                        axes[3].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha)
                        axes[3].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha)
                        axes[3].scatter(cur_ship1, cur_ship2, color='r', label="Ship")
                        axes[3].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                        try:
                            prev_x = data['X'][dtidx-1]
                            kalman_cur12 = prev_x[kfidx]
                            kalman_cur22 = prev_x[kfidx+n_data]
                            kalman_curN2 = kalman_cur12
                            kalman_curE2 = kalman_cur22
                            kalman_cur1_22 = kalman_cur12*np.cos(ais_data["Phi1"][grid0][grid1])+kalman_cur22*np.sin(ais_data["Phi1"][grid0][grid1])
                            kalman_cur22 = kalman_cur12*np.cos(ais_data["Phi2"][grid0][grid1])+kalman_cur22*np.sin(ais_data["Phi2"][grid0][grid1])
                            kalman_cur12 = kalman_cur1_22

                            jcope_cur12 = data['JCOPE'][dtidx-1][kfidx]
                            jcope_cur22 = data['JCOPE'][dtidx-1][kfidx+n_data]
                            jcope_cur1_2 = jcope_cur12*np.cos(ais_data["Phi1"][grid0][grid1])+jcope_cur22*np.sin(ais_data["Phi1"][grid0][grid1])
                            jcope_cur2 = jcope_cur12*np.cos(ais_data["Phi2"][grid0][grid1])+jcope_cur22*np.sin(ais_data["Phi2"][grid0][grid1])
                            jcope_cur1 = jcope_cur1_2
                            axes[3].scatter(jcope_cur1, jcope_cur2, color="pink", alpha=alpha, label="Prev JCOPE")
                            axes[0].scatter(jcope_cur12, jcope_cur22, color="pink", alpha=alpha)
                            axes[3].scatter(kalman_cur12, kalman_cur22, color='black', alpha=alpha, label="Prev Kalman")
                            axes[0].scatter(kalman_curN2, kalman_curE2, color='black', alpha=alpha)
                        except:
                            None

                        def F_mat(jcope0, jcope1, _2N0):
                            a = 0.5
                            F = np.zeros((_2N0+1, _2N0+1), dtype=np.float32)

                            for i in range(_2N0):
                                F[i, i] = a
                                residual = jcope1[i, 0] - a*jcope0[i, 0]
                                F[i, -1] = np.float32(residual)
                            F[_2N0, -1] = np.float32(1.0)
                            # print(f'F:{F}F')
                            return F

                        _2N0 = len(data["JCOPE"][dtidx])-1
                        jcope0 = get_x(jcope_n, jcope_e, dtidx-1, _2N0)
                        jcope1 = get_x(jcope_n, jcope_e, dtidx, _2N0)
                        F = F_mat(jcope0, jcope1, _2N0)
                        x = F @ prev_x
                        x1 = x[kfidx]*np.cos(ais_data["Phi1"][grid0][grid1])+x[kfidx+n_data]*np.sin(ais_data["Phi1"][grid0][grid1])
                        x2 = x[kfidx]*np.cos(ais_data["Phi2"][grid0][grid1])+x[kfidx+n_data]*np.sin(ais_data["Phi2"][grid0][grid1])
                        axes[0].scatter(x[kfidx], x[kfidx+n_data], color='grey', alpha=alpha, label="Kalman-")
                        axes[0].annotate('', xy=(kalman_curN2, kalman_curE2), xytext=(x[kfidx], x[kfidx+n_data]), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        axes[0].annotate('', xy=(x[kfidx], x[kfidx+n_data]), xytext=(kalman_curN, kalman_curE), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        axes[3].scatter(x1, x2, color='grey', alpha=alpha, label="Kalman-")
                        axes[3].annotate('', xy=(kalman_cur12, kalman_cur22), xytext=(x1, x2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        axes[3].annotate('', xy=(x1, x2), xytext=(kalman_cur1, kalman_cur2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))

                        #[axes[axi].set_xlim(-4, 4) for axi in range(len(axes))]
                        #[axes[axi].set_ylim(-4, 4) for axi in range(len(axes))]

                        axes[0].set_title('(N, E): Prev Kalman->Kalman- Kalman-->Kalman')
                        axes[3].set_title('(Cur1, Cur2): Prev Kalman->Kalman- Kalman-->Kalman')
                        fig.legend(loc="lower center", ncol=10, bbox_to_anchor=(0.5, -0.01))
                        K = data['K'][dtidx][kfidx]
                        print(f"K: {np.round(K, 3)}")
                        plt.show()
                        #axes[1].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha)
                        if is_first[1]:
                            #axes[3].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha, label='AIS')
                            is_first[1] = False
                        else:
                            #axes[3].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha)
                            None 
                    if use_ais_removed_bad_mmsi:
                        if ais_cur1_2==ais_cur1_2 and ais_cur2_2==ais_cur2_2:
                            df_a2j.append([ais_cur1_2-jcope_cur1, ais_cur2_2-jcope_cur2])
                            df_a2s.append([ais_cur1_2-curN, ais_cur2_2-curE])
                            df_a2k.append([ais_cur1_2-kalman_cur1, ais_cur2_2-kalman_cur2])

                            #axes[1].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha)
                            if is_first[2]:
                                #axes[3].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha, label='AIS')
                                is_first[2] = False
                            else:
                                #axes[3].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha)
                                None
                    if available_ais:
                        df_sj.append([jcope_cur1-cur_ship1, jcope_cur2-cur_ship2])    
                        df_kj.append([jcope_cur1-kalman_cur1, jcope_cur2-kalman_cur2])    
                        df_ks.append([kalman_cur1-cur_ship1, kalman_cur2-cur_ship2]) 
                        diffs.append([kalman_cur1-cur_ship1, kalman_cur2-cur_ship2])

                        all_data['day'].append([day])
                        all_data['grid'].append([grid0, grid1])
                        all_data['ship'].append([cur_ship1, cur_ship2])
                        all_data['kalman'].append([kalman_cur1, kalman_cur2])
                        all_data['ais'].append([ais_cur1, ais_cur2])
                        if use_ais_removed_bad_mmsi:
                            all_data['ais2'].append([ais_cur1_2, ais_cur2_2])
                        all_data['jcope'].append([jcope_cur1, jcope_cur2])

                        #axes[0].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                        #axes[2].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha)
                        if is_first[0]:
                            #axes[3].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha, label='Kalman')
                            #axes[3].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha, label='JCOPE')
                            None
                        else:
                            #axes[3].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                            #axes[3].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha)
                            None

                        for ax in axes:
                            if ax==axes[-1] and is_first[0]:
                                #ax.scatter(cur_ship1, cur_ship2, color='r', label='Shiplog')
                                None
                            else:
                                #ax.scatter(cur_ship1, cur_ship2, color='r')
                                None

                        is_first[0] = False
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
    print(f'a2j: {np.mean(np.abs(df_a2j))}, ndata={len(df_a2j)}')
    print(f'a2k: {np.mean(np.abs(df_a2k))}, ndata={len(df_a2k)}')
    print(f'a2s: {np.mean(np.abs(df_a2s))}, ndata={len(df_a2s)}')
    print(f'kj: {np.mean(np.abs(df_kj))}, ndata={len(df_kj)}')
    print(f'sj: {np.mean(np.abs(df_sj))}, ndata={len(df_sj)}')
    print(f'ks: {np.mean(np.abs(df_ks))}, ndata={len(df_ks)}')

    pklfilename = osp.join(path_log, 'analysis_data.pkl')
    pickle_dump(all_data, pklfilename)
    print(f'Saved in {pklfilename}')

    for target_ship in target_ships:
        a = np.mean(diff_ship[target_ship][0])
        b = np.mean(diff_ship[target_ship][1])
        c = diff_ship[target_ship][2]
        print(f'{target_ship}: {round(np.mean(a), 2)}, {round(np.mean(b), 2)}, {c}')

    if False:
        for ax in axes:
            ax.set_xlabel('N')
            ax.set_ylabel('E')
        axes[0].set_title('Ploted Kalman results and Shiplog')
        axes[1].set_title('Ploted AIS and Shiplog')
        axes[2].set_title('Ploted JCOPE and Shiplog')
        axes[3].set_title('Ploted Kalman results, AIS, JCOPE and Shiplog')
        fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        plt.show()
    else:
        axes[0].set_title('AIS->Kalman')
        axes[1].set_title('JCOPE->Kalman')
        axes[2].set_title('Kalman->Ship')
        plt.show()
        plt.close()


    return ['ais-jcope', 'ais-kalman', 'ais-ship', 'kalman-jcope', 'ship-jcope', 'kalman-ship'],\
           [np.mean(np.abs(df_aj)), np.mean(np.abs(df_ak)),np.mean(np.abs(df_as)),np.mean(np.abs(df_kj)),np.mean(np.abs(df_sj)),np.mean(np.abs(df_ks)),],\
           [len(df_aj), len(df_ak),len(df_as),len(df_kj),len(df_sj),len(df_ks)],\
            all_data

def make_analysis_pkl(kl, al, al2):
    data = kl.load_kalmanLog_day(1, s, keys=['X'])

    files = os.listdir(path_ship)
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))]

    df_aj = []
    df_as = []
    df_ak = []
    df_a2j = []
    df_a2s = []
    df_a2k = []
    df_sj = []
    df_kj = []
    df_ks = []
    dtidxs = []
    grids = []
    diff_ship = {}

    keys = ['day', 'grid', 'ship', 'kalman', 'ais', 'ais2','jcope', 'lambda','original_lambda']
    all_data = {}
    for key in keys:
        all_data[key] = []

    for target_ship in target_ships:
        diff_ship[target_ship] = [[], [], 0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    colors = {'ship': 'r', 
              'kalman' : 'b', 
              'ais' : 'g', 
              'ais2' : 'gray', 
              'jcope' : 'purple'}

    is_first = [True, True, True]
    n_data = -1
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
            #path_log = osp.join(path_ship, target_ship, '2015')
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
                ais_data = al.load_ais_dtidx(dtidx2, keys=["N", "E", "Lambda1", "Lambda2"])
                if use_ais_removed_bad_mmsi:
                    ais_data2 = al2.load_ais_dtidx(int(dtidx2), keys=["n", "e", "lambda1", "lambda2"])
                #print(f"dtidx: {dtidx2}")
                if type(ais_data["N"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    #print(f"N shape: {dataN.shape}")
                    dataN = ais_data["N"]
                    ais_cur1 = dataN[grid0][grid1]
                    dataE = ais_data["E"]
                    ais_cur2 = dataE[grid0][grid1]

                    if use_ais_removed_bad_mmsi:
                        dataN = ais_data2["n"]
                        ais_cur1_2 = dataN[grid0][grid1]
                        dataE = ais_data2["e"]
                        ais_cur2_2 = dataE[grid0][grid1]
                        lambda1_2 = ais_data2["lambda1"]
                        lambda2_2 = ais_data2["lambda2"]
                    

                    lambda1 = ais_data["Lambda1"]
                    lambda1 = lambda1[grid0][grid1]
                    lambda2 = ais_data["Lambda2"]
                    lambda2 = lambda2[grid0][grid1]
                    
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                    elif not lambda2 > 8000:
                        print(f"lambda2 is low {lambda2}, cur: {ais_cur1}, {ais_cur2}")

                    alpha = 0.8
                    #if lambda2 > 8000 and ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                    if ais_cur1==ais_cur1 and ais_cur2==ais_cur2:
                        df_aj.append([ais_cur1-jcope_cur1, ais_cur2-jcope_cur2])
                        df_as.append([ais_cur1-curN, ais_cur2-curE])
                        df_ak.append([ais_cur1-kalman_cur1, ais_cur2-kalman_cur2])

                        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
                        axes[0].annotate('', xy=(ais_cur1, ais_cur2), xytext=(kalman_cur1, kalman_cur2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        axes[1].annotate('', xy=(jcope_cur1, jcope_cur2), xytext=(kalman_cur1, kalman_cur2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        axes[3].annotate('', xy=(ais_cur1, ais_cur2), xytext=(kalman_cur1, kalman_cur2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        axes[3].annotate('', xy=(jcope_cur1, jcope_cur2), xytext=(kalman_cur1, kalman_cur2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))

                        axes[0].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha, label="AIS")
                        axes[3].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha)
                        axes[1].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha, label="JCOPE")
                        axes[3].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha)
                        axes[2].scatter(curN, curE, color='r', label="Ship")
                        axes[0].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha, label="Kalman")
                        axes[1].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                        axes[2].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                        axes[3].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                        try:
                            kalman_cur12 = data['X'][dtidx-1][kfidx]
                            kalman_cur22 = data['X'][dtidx-1][kfidx+n_data]

                            axes[0].scatter(kalman_cur12, kalman_cur22, color='black', alpha=alpha, label="Prev Kalman")
                            axes[1].scatter(kalman_cur12, kalman_cur22, color='black', alpha=alpha)
                            axes[2].scatter(kalman_cur12, kalman_cur22, color='black', alpha=alpha)
                            axes[3].scatter(kalman_cur12, kalman_cur22, color='black', alpha=alpha)
                            axes[3].annotate('', xy=(kalman_cur12, kalman_cur22), xytext=(kalman_cur1, kalman_cur2), arrowprops=dict(facecolor='blue', arrowstyle='<-'))
                        except:
                            None

                        [axes[axi].set_xlim(-4, 4) for axi in range(len(axes))]
                        [axes[axi].set_ylim(-4, 4) for axi in range(len(axes))]

                        axes[0].set_title('AIS->Kalman')
                        axes[1].set_title('JCOPE->Kalman')
                        axes[2].set_title('Kalman Ship')
                        axes[2].set_title('AIS->Kalman, JCOPE->Kalman, Prev Kalman->Kalman ')
                        fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
                        #plt.show()

                        axes[1].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha)
                        if is_first[1]:
                            axes[3].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha, label='ais')
                            is_first[1] = False
                        else:
                            axes[3].scatter(ais_cur1, ais_cur2, color=colors['ais'], alpha=alpha)
                    if use_ais_removed_bad_mmsi:
                        if ais_cur1_2==ais_cur1_2 and ais_cur2_2==ais_cur2_2:
                            df_a2j.append([ais_cur1_2-jcope_cur1, ais_cur2_2-jcope_cur2])
                            df_a2s.append([ais_cur1_2-curN, ais_cur2_2-curE])
                            df_a2k.append([ais_cur1_2-kalman_cur1, ais_cur2_2-kalman_cur2])

                            axes[1].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha)
                            if is_first[2]:
                                axes[3].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha, label='AIS')
                                is_first[2] = False
                            else:
                                axes[3].scatter(ais_cur1_2, ais_cur2_2, color=colors['ais2'], alpha=alpha)
                    df_sj.append([jcope_cur1-curN, jcope_cur2-curE])    
                    df_kj.append([jcope_cur1-kalman_cur1, jcope_cur2-kalman_cur2])    
                    df_ks.append([kalman_cur1-curN, kalman_cur2-curE]) 
                    diffs.append([kalman_cur1-curN, kalman_cur2-curE])

                    all_data['day'].append([day])
                    all_data['grid'].append([grid0, grid1])
                    all_data['ship'].append([curN, curE])
                    all_data['kalman'].append([kalman_cur1, kalman_cur2])
                    all_data['ais'].append([ais_cur1, ais_cur2])
                    all_data['lambda'].append([lambda1, lambda2])
                    if use_ais_removed_bad_mmsi:
                        all_data['ais2'].append([ais_cur1_2, ais_cur2_2])
                        all_data['original_lambda'].append([lambda1_2, lambda2_2])
                    all_data['jcope'].append([jcope_cur1, jcope_cur2])

                    axes[0].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                    axes[2].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha)
                    if is_first[0]:
                        axes[3].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha, label='Kalman')
                        axes[3].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha, label='JCOPE')
                    else:
                        axes[3].scatter(kalman_cur1, kalman_cur2, color=colors['kalman'], alpha=alpha)
                        axes[3].scatter(jcope_cur1, jcope_cur2, color=colors['jcope'], alpha=alpha)

                    for ax in axes:
                        if ax==axes[-1] and is_first[0]:
                            ax.scatter(curN, curE, color='r', label='Shiplog')
                        else:
                            ax.scatter(curN, curE, color='r')

                    is_first[0] = False
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
    print(f'a2j: {np.mean(np.abs(df_a2j))}, ndata={len(df_a2j)}')
    print(f'a2k: {np.mean(np.abs(df_a2k))}, ndata={len(df_a2k)}')
    print(f'a2s: {np.mean(np.abs(df_a2s))}, ndata={len(df_a2s)}')
    print(f'kj: {np.mean(np.abs(df_kj))}, ndata={len(df_kj)}')
    print(f'sj: {np.mean(np.abs(df_sj))}, ndata={len(df_sj)}')
    print(f'ks: {np.mean(np.abs(df_ks))}, ndata={len(df_ks)}')

    pklfilename = osp.join(path_log, 'analysis_data.pkl')
    pickle_dump(all_data, pklfilename)
    print(f'Saved in {pklfilename}')

    for target_ship in target_ships:
        a = np.mean(diff_ship[target_ship][0])
        b = np.mean(diff_ship[target_ship][1])
        c = diff_ship[target_ship][2]
        print(f'{target_ship}: {round(np.mean(a), 2)}, {round(np.mean(b), 2)}, {c}')

    if False:
        for ax in axes:
            ax.set_xlabel('N')
            ax.set_ylabel('E')
        axes[0].set_title('Ploted Kalman results and Shiplog')
        axes[1].set_title('Ploted AIS and Shiplog')
        axes[2].set_title('Ploted JCOPE and Shiplog')
        axes[3].set_title('Ploted Kalman results, AIS, JCOPE and Shiplog')
        fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        plt.show()
    else:
        plt.close()


    return ['ais-jcope', 'ais-kalman', 'ais-ship', 'kalman-jcope', 'ship-jcope', 'kalman-ship'],\
           [np.mean(np.abs(df_aj)), np.mean(np.abs(df_ak)),np.mean(np.abs(df_as)),np.mean(np.abs(df_kj)),np.mean(np.abs(df_sj)),np.mean(np.abs(df_ks)),],\
           [len(df_aj), len(df_ak),len(df_as),len(df_kj),len(df_sj),len(df_ks)],\
            all_data

def analysis():
    save_env_map()
    print(f'Target kalmanlog : {path_log}')
    kl = KalmanLogLoader(2015, 9, 20)
    kl.set_path(path_log)

    jl = JCOPELoader(year, month)
    jl.load_path(path_jcope)

    if not use_ais_removed_bad_mmsi:
        #ais_outfiles = osp.join(path_ais, 'ais_files')
        #AISLoader = atc.AISLoader
        #al = AISLoader(year, month, ais_outfiles, pkl_path=path_ais)
        #al.set_keys(ais_keys)
        ais_keys = ['N', 'E', "Cur1", "Cur2", "Lambda1", 'Lambda2', "Phi1", "Phi2"]
        from utils.ais_loader import AISLoader
        al = AISLoader(year, month, pool_size, use_ais_remove_bad_mmsi=False)
        al.set_keys(ais_keys)
        al.load_path()
        al2 = None
    else:
        ais_keys = ['N', 'E', "Cur1", "Cur2", "Lambda1", 'Lambda2', "Phi1", "Phi2"]
        from utils.ais_loader import AISLoader
        al = AISLoader(year, month, pool_size, use_ais_remove_bad_mmsi=True)
        al.set_keys(ais_keys)
        al.load_path()

        al2 = AISLoader(year, month, pool_size, use_ais_remove_bad_mmsi=False)
        al2.set_keys(ais_keys)
        al2.load_path()


    if AIS_JCOPE_TEST: 
        #eval_ais(al)
        eval_ais_lambdavec(al)
        #eval_jcope(jl)

    keys = ['kalman-n', 'kalman-e', 'jcope-n', 'jcope-e', 'diff-kalman-jcope-n', 'diff-kalman-jcope-e', 'ais-n', 'ais-e']
    if VISUALIZE:
        save_gif = True
        save_point_graph = True

        m_day = 2 if MAX_DAY==1 else MAX_DAY
        for day in range(1, MAX_DAY+1):
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
    
     
    if EVALUATION:
        if use_pickle:
            failed_load_file = False 
            filepath = osp.join(path_log, 'analysis_data.pkl')
            all_data = pickle_load(filepath)
            eval_kalmanLog(all_data, [0, 0], False)
            #try:
            #    filepath = osp.join(path_log, 'analysis_data.pkl')
            #    all_data = pickle_load(filepath)
            #    eval_kalmanLog(all_data)
            #except:
            #    failed_load_file = True
            #    filepath = osp.join(path_log, 'analysis_data.pkl')
            #    print(f"Failed to load file from {filepath}")

        if not use_pickle or failed_load_file:
            #_, _, _, all_data = make_analysis_pkl(kl, al, al2)
            #_, _, _, all_data = make_analysis_lambdacur_pkl(kl, al, al2, jl)
            _, _, _, all_data = make_lambdacur_pkl(kl, al, al2, jl)
            eval_kalmanLog(all_data)


def eval_ais_lambdavec(al):
    # 船のpathの設定
    files = os.listdir(path_ship)
    # path内にあるファイルの一覧取得
    target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path_ship, f))] 

    # aisと船の偏流値の差の大きさ(|ais-ship|)
    df_as = {   "Dtidx":[],
                "ShipName":[],
                "Cur1":[],
                "Cur2":[],
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
            #path_log = osp.join(path_ship, target_ship, '2015')
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
                #keys = ["Cur1", "Cur2", "Lambda1", "Lambda2", "Phi1", "Phi2"]
                keys = ["cur1", "cur2", "lambda1", "lambda2", "psi1", "psi2"]
                ais_data = al.load_ais_dtidx(dtidx2, keys=keys)
                #print(f"dtidx: {dtidx2}")
                if type(ais_data["cur1"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    #print(f"N shape: {dataN.shape}")
                    ais_cur1 = ais_data["cur1"][grid0][grid1]
                    
                    ais_cur2 = ais_data["cur2"][grid0][grid1]

                    lambda2 = ais_data["lambda2"]
                    lambda2 = lambda2[grid0][grid1]
                    
                    # nanでなく，固有値が一定以上の場合で比較する
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                        continue
                    if not lambda2 > 100:
                        print(f"lambda2 is low {lambda2}, cur: {ais_cur1}, {ais_cur2}")
                        continue
                    else:
                        df_as["Dtidx"].append(dtidx)
                        df_as["ShipName"].append(target_ship)
                        # Phi1, Phi2 is based on North 
                        cur_ship1 = curN*np.cos(ais_data["psi1"][grid0][grid1])+curE*np.sin(ais_data["psi1"][grid0][grid1])
                        df_as["Cur1"].append(ais_cur1-cur_ship1)
                        cur_ship2 = curN*np.cos(ais_data["psi2"][grid0][grid1])+curE*np.sin(ais_data["psi2"][grid0][grid1])
                        df_as["Cur2"].append(ais_cur2-cur_ship2)
                        df_as_grid[0][grid0][grid1] += np.abs(ais_cur1-cur_ship1)
                        df_as_grid[1][grid0][grid1] += np.abs(ais_cur2-cur_ship2)
                        count_grid[grid0][grid1] += 1 
                        print(f'ais: Cur1={ais_cur1}, Cur2={ais_cur2}')
                        print(f'ship: Cur1={cur_ship1}, Cur2={cur_ship2}')

                    # ログの出力
                    if len(df_as["Cur1"])>0:
                        Cur1 = np.array(df_as["Cur1"])
                        Cur2 = np.array(df_as["Cur2"])
                        n_df_data = len(Cur1)
                        print(f'as: N={np.mean(np.abs(Cur1))}, E={np.mean(np.abs(Cur2))}, ndata={n_df_data}')

    # データの整理
    dtidxs = np.array(df_as["Dtidx"])
    shipName = np.array(df_as["ShipName"])
    N = np.array(df_as["Cur1"])
    E = np.array(df_as["Cur2"])

    N = N[~np.isnan(N)]
    E = E[~np.isnan(N)] 
    tf = count_grid>0
    df_as_grid[0][tf] /= count_grid[tf]
    df_as_grid[1][tf] /= count_grid[tf]

    # 性能の結果
    if len(df_as["Cur1"])>0:
        print(f'as: Cur1={np.mean(np.abs(N))}, Cur2={np.mean(np.abs(E))}, ndata={len(df_as["Cur1"])}')

    if False:
        # 時間の違いの性能の結果
        plt.figure()
        plt.scatter(dtidxs, N, label="Cur1")
        plt.scatter(dtidxs, E, label="Cur2")
        plt.legend()
        plt.xlabel('Dtidx')
        plt.ylabel('Diff AIS and Ship')
        savepath = "./logs/analysis_diff-ais_lambdacur-ship_dtidx.png"
        plt.savefig(savepath)

        # 船の違いの性能の結果
        plt.figure(figsize=(16, 16))
        N2 = []
        E2 = []
        for target_ship in target_ships:
            tf = shipName == target_ship
            if np.sum(tf)>0:
                N2.append(np.mean(np.abs(N[tf])))
                E2.append(np.mean(np.abs(E[tf])))
            else:
                N2.append(np.nan)
                E2.append(np.nan)
        x = np.arange(len(target_ships))
        width = 0.4
        plt.bar(x-width/2, N2, color='b', label="North", tick_label=target_ships)
        plt.bar(x+width/2, E2, color='r', label="East", tick_label=target_ships)
        plt.xlabel('Ship Name')
        plt.ylabel('Diff AIS and Ship')
        plt.legend()
        savepath = "./logs/analysis_diff-ais_lambdacur-ship_shipname.png"
        plt.savefig(savepath)

        # 船ごとの使用したデータ数
        plt.figure(figsize=(16, 16))
        counts = []
        for target_ship in target_ships:
            tf = shipName == target_ship
            counts.append(np.sum(tf))
        plt.bar(target_ships, counts, color='b')
        plt.xlabel('Ship Name')
        plt.ylabel('Number of Data')
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
        savepath = "./logs/analysis_diff-ais_lambdacur-ship_grid.png"
        plt.savefig(savepath)
def eval_ais(al):
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
            #path_log = osp.join(path_ship, target_ship, '2015')
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
                ais_data = al.load_ais_dtidx(dtidx2, keys=["N", "E", "Lambda2"])
                #print(f"dtidx: {dtidx2}")
                if type(ais_data["N"])!=type([]):
                    #print(f"grid0: {grid0}, grid1 {grid1}")
                    dataN = ais_data["N"]
                    #print(f"N shape: {dataN.shape}")
                    ais_cur1 = dataN[grid0][grid1]
                    
                    dataE = ais_data["E"]
                    ais_cur2 = dataE[grid0][grid1]

                    lambda2 = ais_data["Lambda2"]
                    lambda2 = lambda2[grid0][grid1]
                    
                    # nanでなく，固有値が一定以上の場合で比較する
                    if not (ais_cur1==ais_cur1 and ais_cur2==ais_cur2):
                        print(f"ais is Nan")
                        continue
                    if not lambda2 > 8000:
                        print(f"lambda2 is low {lambda2}, cur: {ais_cur1}, {ais_cur2}")
                        continue
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
                    if len(df_as["N"])>0:
                        N = np.array(df_as["N"])
                        E = np.array(df_as["E"])
                        n_df_data = len(N)
                        print(f'as: N={np.mean(np.abs(N))}, E={np.mean(np.abs(E))}, ndata={n_df_data}')

    # データの整理
    dtidxs = np.array(df_as["Dtidx"])
    shipName = np.array(df_as["ShipName"])
    N = np.array(df_as["N"])
    E = np.array(df_as["E"])

    N = N[~np.isnan(N)]
    E = E[~np.isnan(N)] 
    tf = count_grid>0
    df_as_grid[0][tf] /= count_grid[tf]
    df_as_grid[1][tf] /= count_grid[tf]

    # 性能の結果
    if len(df_as["N"])>0:
        print(f'as: N={np.mean(np.abs(N))}, E={np.mean(np.abs(E))}, ndata={len(df_as["N"])}')

    # 時間の違いの性能の結果
    plt.figure()
    plt.scatter(dtidxs, N, label="North")
    plt.scatter(dtidxs, E, label="East")
    plt.legend()
    plt.xlabel('Dtidx')
    plt.ylabel('Diff AIS and Ship')
    savepath = "./logs/analysis_diff-ais-ship_dtidx.png"
    plt.savefig(savepath)

    # 船の違いの性能の結果
    plt.figure(figsize=(16, 16))
    N2 = []
    E2 = []
    for target_ship in target_ships:
        tf = shipName == target_ship
        if np.sum(tf)>0:
            N2.append(np.mean(np.abs(N[tf])))
            E2.append(np.mean(np.abs(E[tf])))
        else:
            N2.append(np.nan)
            E2.append(np.nan)
    x = np.arange(len(target_ships))
    width = 0.4
    plt.bar(x-width/2, N2, color='b', label="North", tick_label=target_ships)
    plt.bar(x+width/2, E2, color='r', label="East", tick_label=target_ships)
    plt.xlabel('Ship Name')
    plt.ylabel('Diff AIS and Ship')
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
    plt.xlabel('Ship Name')
    plt.ylabel('Number of Data')
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

def eval_jcope(jl):
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
        jcope_n, jcope_e = jl.load_jcope_day(day)
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
            #path_log = osp.join(path_ship, target_ship, '2015')
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
                #print(f"dtidx: {dtidx2}")
                if dtidx in jcope_n.keys() and dtidx in jcope_e.keys():
                    N = jcope_n[dtidx][grid0][grid1]
                    E = jcope_e[dtidx][grid0][grid1]
                    
                    if not (N==N and E==E):
                        print(f"ais is Nan")
                        continue
                    else:
                        df_as["Dtidx"].append(dtidx)
                        df_as["ShipName"].append(target_ship)
                        df_as["N"].append(N-curN)
                        df_as["E"].append(E-curE)
                        df_as_grid[0][grid0][grid1] += np.abs(N-curN)
                        df_as_grid[1][grid0][grid1] += np.abs(E-curE)
                        count_grid[grid0][grid1] += 1 
                        print(f'ais: N={N}, E={E}')
                        print(f'ship: N={curN}, E={curE}')

                    # ログの出力
                    if len(df_as)>0:
                        N = np.array(df_as["N"])
                        E = np.array(df_as["E"])
                        n_df_data = len(N)
                        print(f'js: N={np.mean(np.abs(N))}, E={np.mean(np.abs(E))}, ndata={n_df_data}')

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
    plt.xlabel(f'Dtidx')
    plt.ylabel(f'Diff AIS and Ship')
    plt.legend()
    savepath = "./logs/analysis_diff-jcope-ship_dtidx.png"
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
    plt.xlabel(f'Ship Name')
    plt.ylabel(f'Diff AIS and Ship')
    plt.legend()
    savepath = "./logs/analysis_diff-jcope-ship_shipname.png"
    plt.savefig(savepath)

    # 船ごとの使用したデータ数
    plt.figure(figsize=(16, 16))
    counts = []
    for target_ship in target_ships:
        tf = shipName == target_ship
        counts.append(np.sum(tf))
    plt.bar(target_ships, counts, color='b')
    plt.xlabel(f'Ship Name')
    plt.ylabel(f'Number of data')
    savepath = "./logs/analysis_shipDataNum.png"
    plt.savefig(savepath)

    # 座標の違いの性能の結果
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    im = axes[0].imshow(df_as_grid[0])
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(df_as_grid[1])
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(count_grid)
    fig.colorbar(im, ax=axes[2])

    axes[0].set_title("North")
    axes[1].set_title("East")
    axes[2].set_title("Num of Data")
    savepath = "./logs/analysis_diff-jcope-ship_grid.png"
    plt.savefig(savepath)


if __name__ == '__main__':
    analysis()
    