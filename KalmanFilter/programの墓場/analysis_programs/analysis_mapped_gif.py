import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
import seaborn as sns
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from Ais4ToCurForKalmanTime import ais4
import Ais4ToCurForKalmanTime as atc
from utils import * 

lat00 = int(map_pooled_size[0]-kurosio_latidx_range1[0]/2)
lat11 = int(map_pooled_size[0]-kurosio_latidx_range2[1]/2)
lon0 = int(map_pooled_size[1]-kurosio_lonidx_range[0]/2)
lon1 = int(map_pooled_size[1]-kurosio_lonidx_range[1]/2)
print(f'lat00 {lat00}')
print(f'lat11 {lat11}')
print(f'lon0 {lon0}')
print(f'lon1 {lon1}')

year = 2015
month = 9
n_day = nday_month(month) - 1
n_hour = 24 #24
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True

print(f'Loading data')

kl = KalmanLogLoader(2015, 9, 20)
log_path = r"E:\shunsukeE\result\kalman-west-kurosio-Q001Q01"
path_ais = r'E:\shunsukeE\data\ais\1509-ais4s-pkls'
kalman_keys = ['X', 'JCOPE', 'Z']
kl.set_path(log_path)

ais_keys = ['n', 'e']
al = atc.AISLoader(year, month, osp.join(path_ais, 'ais_files'))
al.set_keys(ais_keys)

def make_path(p1, p2, extension=None):
    path = osp.join(p1, p2)
    if extension:
        path = path+extension
    return path

class GifMaker():
    def __init__(self):
        self.datas = []

    def add_data(self, d):
        self.datas += [d]

    def add_datas(self, ds):
        self.datas += ds

    def make(self, titles=None, folder="./", file_name="Non", save=True, show=False):
        def make_heatmap(i):
            ax.cla()
            if titles:
                ax.set_title(titles[i])
            else:
                ax.set_title("Iteration="+str(i))
            data = np.array(self.datas[i])
            sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax)
            ax.set_aspect('equal', adjustable='box')
        #fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        fms = len(self.datas) 
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
        ani = animation.FuncAnimation(fig=fig, func=make_heatmap, frames=fms, interval=500, blit=False)
        if save:
            file_path = make_path(folder, file_name, extension=".gif")
            ani.save(file_path, writer="pillow")
        if show:
            plt.show() 
        plt.close()

    def reset(self):
        plt.close()
        self.datas = []


for s in range(0, 1):
    kalman_n_maps = []
    kalman_e_maps = []
    jcope_n_maps = []
    jcope_e_maps = []
    kalman_jcope_n_maps = []
    kalman_jcope_e_maps = []
    ais_n_maps = []
    ais_e_maps = []
    connected_n_maps = []
    connected_e_maps = []
    connected_dif_maps = []
    titles = []
    for day in tqdm.tqdm(range(1, n_day+1)):
        print(f's:{s}, day:{day}')
        # TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']
        if is_light:
            TF = np.array([True] * 3789)
        else:
            TF = np.array([True] * 9808)

        start_hour = 0 if day == 1 else 1
        for hour in range(start_hour, n_hour):
            titles.append(f'{month}/{day} {hour}:00')
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue

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

            lat00 = int(map_pooled_size[0]-kurosio_latidx_range1[0]/2)
            lat11 = int(map_pooled_size[0]-kurosio_latidx_range2[1]/2)
            lon0 = int(map_pooled_size[1]-kurosio_lonidx_range[0]/2)
            lon1 = int(map_pooled_size[1]-kurosio_lonidx_range[1]/2)
            dif = 3 

            n_map = n_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            e_map = e_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            jn_map = jn_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            je_map = je_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            kjn_map = kjn_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            kje_map = kje_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            an_map = an_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T
            ae_map = ae_map[lat11-dif:lat00+dif].T[lon1-dif:lon0+dif].T


            kalman_n_maps.append(n_map)
            kalman_e_maps.append(e_map)

            jcope_n_maps.append(jn_map) 
            jcope_e_maps.append(je_map)

            kalman_jcope_n_maps.append(kjn_map)
            kalman_jcope_e_maps.append(kje_map)

            ais_n_maps.append(an_map)
            ais_e_maps.append(ae_map)

            c_n_map = np.concatenate([an_map, jn_map, n_map])
            connected_n_maps.append(c_n_map)
            c_e_map = np.concatenate([ae_map, je_map, e_map])
            connected_e_maps.append(c_e_map)
            c_dif_map = np.concatenate([kjn_map, kje_map])
            connected_dif_maps.append(c_dif_map)
        
    gif_maker = GifMaker()

    #datas = [kalman_n_maps, kalman_e_maps, jcope_n_maps, jcope_e_maps, kalman_jcope_n_maps, kalman_jcope_e_maps, ais_n_maps, ais_e_maps]
    #keys = ['kalman_n', 'kalman_e', 'jcope_n', 'jcope_e', 'kalman-jcope_n', 'kalman-jcope_e', 'ais_n', 'ais_e']
    #datas  = [jcope_n_maps, jcope_e_maps]
    #keys = ['jcope_n', 'jcope_e']
    datas  = [connected_n_maps, connected_e_maps, connected_dif_maps]
    keys = ['N', 'E', 'NE']

    for i in tqdm.tqdm(range(len(datas))):
        titles2 = [ t + f" ({keys[i]})" for t in titles]
        fname = f'{keys[i]}{year}{month:02}'
        gif_maker.add_datas(datas[i])
        gif_maker.make(titles=titles2, folder=log_path, file_name=fname)
        gif_maker.reset()
