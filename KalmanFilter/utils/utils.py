import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import randn
import seaborn as sns
import tqdm
import json
import math
import os.path as osp
import pickle as pkl
import inspect
import matplotlib.animation as animation

### Functions for data processing ###
# 移動平均
def mean_pre_nex(data, pre=5, nex=5):
    m_data = np.array(data)
    ave_data = []
    for i in np.arange(len(m_data)):
        p_pre = i-pre if i-pre>=0 else 0
        p_nex = i+nex if i+nex<len(m_data) else len(m_data)-1
        ave_data += [np.mean(m_data[p_pre:p_nex])]
    return ave_data  

def extract_datas(datas, row_range, col_range):
    n_data = len(datas)

    res = []
    for n in range(n_data):
        data = np.array(datas[n])
        data = data[row_range[0]:row_range[1]].T[col_range[0]:col_range[1]].T
        res.append(data)
    return res

# loggerで保存されたjsonファイルの読み込みに使う
def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

### Other programs ###
# importされたファイル一覧の取得
def get_imported_module_names():
    names = []
    stack = inspect.stack()
    for s in stack:
        for i in range(len(s)):
            m = inspect.getmodule(s[i])
            if m:
                filename = osp.basename(m.__file__)
                if filename not in names:
                    names.append(filename)
    print(names)
    return names

# importされたファイルの中にkeyが含まれているかを返す
def is_imported_file(keys):
    imported_filename = get_imported_module_names()
    is_imported = {}
    for key in keys:
        is_imported[key] = False
    for name in imported_filename:
        for key in is_imported.keys():
            if name.find(key)!=-1:
                is_imported[key] = True
    return is_imported


# pickle保存
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pkl.dump(obj,f)

# pickle読み込み
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pkl.load(f)
        return data 

from sklearn.neighbors import KernelDensity
def calc_kde(prob_datas, num_agent=1000):
    datas = np.array(prob_datas)
    d_shape = datas.shape
    assert len(d_shape)==2, f'shape error in kde plot {d_shape}'

    n_data = (datas*1000).astype(np.int64)

    # 2次元の確率データの例（仮のデータ）
    #data = np.random.randn(1000, 2)  # 1000個の2次元データポイント
    data = []
    for r in range(d_shape[0]):
        for c in range(d_shape[1]):
            data += [[r, c] for i in range(n_data[r, c])]
    data = np.array(data)

    n_data = np.zeros(d_shape)

    # カーネル密度推定を使用してPDFを推定
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)

    # 等高線プロットの準備
    x = y = np.array([i for i in range(d_shape[0])]) 
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # 確率密度関数の計算
    Z = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)
    Z2 = []
    for i in range(len(Z)):
        Z2.append(Z[len(Z)-i-1])
    Z = Z2
    return X, Y, Z, n_data


### for nmri programs ###
def get_project_name():
    project_names = ['analysis', 'kalman_filter', 'Ais4', 'test']
    res = is_imported_file(project_names)
    sum_res = 0
    detected_project = []
    for key in res.keys():
        sum_res += res[key]
        if res[key]:
            detected_project.append(key)
    if sum_res==0:
        print(f'No module is detected.')
        print(res)
        return detected_project 
    if sum_res>1:
        print(f'Two or more modules are detected.')
        print(res)
        return detected_project 
    else:
        print(res)
        return detected_project 

def dms_to_deg(dms):
    dms_value = float(dms)
    degrees = int(dms_value)
    minutes = int((dms_value - degrees) * 100)
    seconds = ((dms_value - degrees) * 100 - minutes) * 100
    decimale = degrees + (minutes / 60) + (seconds / 3600)
    return decimale/100

def azimuth_to_radian(azimuth):
    radian = 360 - azimuth
    radian += 90
    radian = radian*np.pi/180
    return radian

def latlon_to_mesh(lat, lon, deg_per_mesh, map_size, latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    grid0 = map_size[0] - int((lat-latlon_range[0])/deg_per_mesh)
    grid1 = int((lon-latlon_range[1])/deg_per_mesh)
    if grid0<0 or grid0>map_size[0]: return [-1, -1]
    if grid1<0 or grid1>map_size[1]: return [-1, -1]
    return [grid0, grid1]

def mesh_to_latlon(grid0, grid1, deg_per_mesh, map_size, latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    lat = (-grid0+map_size[0])*deg_per_mesh+latlon_range[0] 
    lon = grid1*deg_per_mesh+latlon_range[1] 
    return [lat, lon]

def latlon_to_mesh_df(lat, lon, deg_per_mesh, size, latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    grid0 = size[0] - ((lon-latlon_range[1]).astype(int)/deg_per_mesh)
    grid1 = ((lat-latlon_range[0]).astype(int)/deg_per_mesh)
    grid0[grid0 < 0] = -1
    grid0[grid0 > size[0]] = -1
    grid1[grid1 < 0] = -1
    grid1[grid1 > size[1]] = -1
    return grid0, grid1

def aisidx_to_rallon(lat_idx, lon_idx, deg_per_mesh):
    return [lat_idx, lon_idx]*deg_per_mesh

def plot_target_points_mesh(grids, base_map, linewidth=1, half_size=50):
    """ 
    About function
        指定座標の周辺のプロット
    examples 
        grids = [[10, 20], [15, 25]]
        base_map = np.ones(map_size) * nan_map
    """
    a = np.array(base_map)
    size = a.shape
    for grid0, grid1 in grids:
        for i in range(linewidth):
            #a[grid0+i][grid1-i:grid1+i] = [-2 for _ in range(2*i)]
            #a[grid0-i][grid1-i:grid1+i] = [-2 for _ in range(2*i)]
            a[grid0+i][:] = -2
            a[grid0-i][:] = -2
            a[:, grid1+i] = -2
            a[:, grid1-i] = -2
    grids_np = np.array(grids)
    grid0_min = np.min(grids_np.t[0])
    grid1_min = np.min(grids_np.t[1])
    grid0_max = np.max(grids_np.t[0])
    grid1_max = np.max(grids_np.t[1])
    box0 = [grid0_min-half_size, grid0_max+half_size]
    if box0[0] < 0:
        box0[0] = 0
    if size[0]<box0[1]:
        box0[1] = size[0]
    box1 = [grid1_min-half_size, grid1_max+half_size]
    if box1[0] < 0:
        box1[0] = 0
    if size[1]<box1[1]:
        box1[1] = size[1]
    b = a[box0[0]:box0[1]]
    b = b[:, box1[0]:box1[1]]
    sns.heatmap(b)
    plt.show()

def haversine_distance(lat1, lon1, lat2, lon2):
    # radius of the earth in kilometers
    earth_radius = 6371.0

    # convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c * 1000

    return distance

def nday_month(month):
    if (month == 2): return 28
    if (month < 8 and month % 2 == 0): return 30
    if (month >= 8 and month % 2 != 0): return 30
    return 31

def date_to_dtidx(base_dt, target_dt):
    # 時間の整理　dtidx: 0時からの経過時間，dtidx_minute:0時0分からの経過分
    idx2 = target_dt - base_dt
    #return int(idx2.days*24*60 + idx2.seconds / (60)) #minutes
    return int(idx2.days*24 + idx2.seconds / (60*60))

def dtidx_to_date(base_dt, hours):
    # hours = dtidx
    # 指定された時間数をdatetime.timedeltaオブジェクトとして作成
    delta = datetime.timedelta(hours=hours)
    # 指定された時間数分後の日時を計算
    target_dt = base_dt + delta
    return target_dt
    
def dummy_jcope_data(x0, dx, count, noise_factor, size=[10, 10]):
    data = []
    for i in range(count):
        data.append([x0+ dx*i + randn()*noise_factor for nm in range(size[0]+size[1])])
        data[i].append(0)
    return data

# dummpy ais data
# data: 1*(n+m)
def dummy_ais_data(x0, dx, omega, count, noise_factor, size=[10, 10]):
    xy = []
    sum_w = []
    for i in range(count):
        xy.append([x0+ dx*i + randn()*noise_factor for nm in range(size[0]+size[1])])
        sum_w.append([1/(2*noise_factor)  for nm in range(size[0]+size[1])])
        xy[i].append(1)
        sum_w[i].append(0.00001)
    return xy, sum_w

# dummpy ais data
# data: 1*(n+m)
def dummy_ais_data2(x0, dx, omega, count, noise_factor, size=[10, 10]):
    v = []
    theta = []
    sum_w = []
    for i in range(count):
        v.append([x0+ dx*i + randn()*noise_factor for nm in range(size[0]+size[1])])
        theta.append([x0+ omega*i + randn()*noise_factor for nm in range(size[0]+size[1])])
        sum_w.append([1/(2*noise_factor)  for nm in range(size[0]+size[1])])
        v[i].append(1)
        theta[i].append(0)
        sum_w[i].append(0)
    return v, theta, sum_w