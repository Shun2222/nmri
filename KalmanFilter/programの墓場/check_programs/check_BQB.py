import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import seaborn as sns
import datetime
from utils import *
from tqdm import tqdm
sns.set(font_scale=4)

year = 2015
month = 9
n_day = 15 #nday_month(month) - 1
n_hour = 12 
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
is_light = True

print(f'Loading data')

log_pathes = [r"E:\shunsukeE\result\kalman-Ptest-Q0.01", r"E:\shunsukeE\result\kalman-Ptest2-Q0.01"]
labels = ['use B', 'not use B']

day = 1
i = 1400
plt.figure(figsize=(24, 24))
plt.title(f'P {month:02}{day:02}{2:02}-{24:02}')
for idx, log_path in enumerate(log_pathes):
    grid0, grid1 = kurosio_grid_pooled[i]
    values = []
    for t in tqdm(range(2, 6)):
        pminus = pkl.load(open(fr'{log_path}/saverPminus{year}{month:02}{day:02}{t:02}.pkl', 'rb'))
        v = pminus[i][i+1]
        values.append(v)

        p = pkl.load(open(fr'{log_path}/saverP{year}{month:02}{day:02}{t:02}-v1.pkl', 'rb'))
        v = p[i][i+1]
        values.append(v)

        p = pkl.load(open(fr'{log_path}/saverP{year}{month:02}{day:02}{t:02}-v2.pkl', 'rb'))
        v = p[i][i+1]
        values.append(v)
    plt.plot(np.arange(len(values))/3+2, values, label=labels[idx])
path = f'Images/Pminus-plot{month:02}{day:02}{t:02}.png'
plt.legend()
plt.savefig(path)
plt.close()
print(f'Saved {path}')

# for t in range(2, 24):
#     pminus = pkl.load(open(fr'{log_path}/saverPminus{year}{month:02}{day:02}{t:02}.pkl', 'rb'))
#     p_map = kurosio_vec_to_map_pooled(pminus[i], nan_map_pooled) * nan_map_pooled
#     p_map = p_map[grid0-5:grid0+5]
#     p_map = p_map.T[grid1-5:grid1+5]
#     p_map = p_map.T
#     print(p_map)
#     #p_map = pminus
#     plt.figure(figsize=(24, 24))
#     plt.title(f'P {month:02}{day:02}{t:02}')
#     sns.heatmap(p_map, cbar=True)
#     path = f'Images/Pminus{month:02}{day:02}{t:02}.png'
#     plt.savefig(path)
#     plt.close()
#     print(f'Saved {path}')

#     # target_name = 'v1'
#     # p = pkl.load(open(fr'{log_path}/saverP{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'rb'))
#     # p_map = kurosio_vec_to_map_pooled(p[i], nan_map_pooled) * nan_map_pooled
#     # #p_map = p
#     # plt.figure(figsize=(24, 24))
#     # plt.title(f'P {month:02}{day:02}{t:02}')
#     # sns.heatmap(p_map, cbar=True)
#     # path = f'Images/Pv1_{month:02}{day:02}{t:02}.png'
#     # plt.savefig(path)
#     # plt.close()
#     # print(f'Saved {path}')

#     # target_name = 'v2'
#     # p = pkl.load(open(fr'{log_path}/saverP{year}{month:02}{day:02}{t:02}-{target_name}.pkl', 'rb'))
#     # p_map = kurosio_vec_to_map_pooled(p[i], nan_map_pooled) * nan_map_pooled
#     # #p_map = p
#     # plt.figure(figsize=(24, 24))
#     # plt.figure(figsize=(24, 24))
#     # plt.title(f'Pv2 {month:02}{day:02}{t:02}')
#     # sns.heatmap(p_map, cbar=True)
#     # path = f'Images/Pv2_{month:02}{day:02}{t:02}.png'
#     # plt.savefig(path)
#     # plt.close()
#     # print(f'Saved {path}')
