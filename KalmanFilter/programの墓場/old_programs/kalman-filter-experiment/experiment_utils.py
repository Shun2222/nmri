from experiment_kf_params import *

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
import tqdm
import math
import pickle as pkl


jcope = pd.read_csv(r"E:\shunsukeE\data\eas2\nan_map.csv", encoding="cp932",)
nan_map = np.array(jcope.values)*0+1

def average_pooling(image, pool_size=(2, 2), stride=None):
    if stride is None:
        stride = pool_size
    height, width = image.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride
    out_height = (height - pool_height) // stride_height + 1
    out_width = (width - pool_width) // stride_width + 1
    pooled = np.zeros((out_height, out_width))
    for y in range(0, out_height):
        for x in range(0, out_width):
            y_start = y * stride_height
            y_end = y_start + pool_height
            x_start = x * stride_width
            x_end = x_start + pool_width
            pooled[y, x] = np.mean(image[y_start:y_end].T[x_start:x_end])
    return pooled

nan_map_pooled = average_pooling(nan_map[:map_size_ais[0]], pool_size=(pool_size, pool_size))

def kurosio(latidx, lonidx):
    dx = kurosio_lonidx_range[1] - kurosio_lonidx_range[0]
    dy1 = kurosio_latidx_range2[0] - kurosio_latidx_range1[0]
    a1 = dy1/dx
    b1 = kurosio_latidx_range1[0] - a1*kurosio_lonidx_range[0]
    dy2 = kurosio_latidx_range2[1] - kurosio_latidx_range1[1]
    a2 = dy2/dx
    b2 = kurosio_latidx_range1[1] - a2*kurosio_lonidx_range[0]
    
    if(lonidx<kurosio_lonidx_range[0] or lonidx>kurosio_lonidx_range[1]):
        return False
    if(a1*lonidx+b1 > latidx or a2*lonidx+b2 < latidx):
        return False
    return True

def kurosio_pooled(latidx, lonidx):
    lon0 = int(kurosio_lonidx_range[0]/pool_size)
    lon1 = int(kurosio_lonidx_range[1]/pool_size)
    lat00 = int(kurosio_latidx_range1[0]/pool_size)
    lat01 = int(kurosio_latidx_range1[1]/pool_size)
    lat10 = int(kurosio_latidx_range2[0]/pool_size)
    lat11 = int(kurosio_latidx_range2[1]/pool_size)

    dx = lon1 - lon0 
    dy1 = lat10 - lat00 
    a1 = dy1/dx
    b1 = lat00 - a1*lon0
    dy2 = lat11 - lat01 
    a2 = dy2/dx
    b2 = lat01 - a2*lon0

    if(lonidx<lon0 or lonidx>lon1):
        return False
    #if( lat00 > latidx or lat11 < latidx):
    #    return False
    if(a1*lonidx+b1 > latidx or a2*lonidx+b2 < latidx):
        return False
    return True 

def kurosio_filter(data, marine_map):
    if data==pd.DataFrame:
        data = data.values
    assert type(data)==np.ndarray
    
    res = []
    for i in range(map_size_ais[0]):
        for j in range(map_size_ais[1]):
            if kurosio(map_size_ais[0]-i, j) and marine_map[i][j]==marine_map[i][j]:
                res.append(data[i][j])
    return np.array(res)

def kurosio_filter_pooled(data, marine_map_pooled, is_pooled=False):
    if data==pd.DataFrame:
        data = data.values
    assert type(data)==np.ndarray
    data = data[:map_size_ais[0]]
    if not is_pooled:
        data = average_pooling(data, pool_size=(pool_size, pool_size))

    res = []
    for i in range(map_pooled_size[0]):
        for j in range(map_pooled_size[1]):
            if kurosio_pooled(map_pooled_size[0]-i, j) and marine_map_pooled[i][j]==marine_map_pooled[i][j]:
                res.append(data[i][j])
    return np.array(res)

def kurosio_vec_to_map(vec, marine_map):
    res = np.zeros(marine_map.shape)
    
    n = 0
    for i in range(map_size_ais[0]):
        for j in range(map_size_ais[1]):
            if kurosio(map_size_ais[0]-i, j) and marine_map[i][j]==marine_map[i][j]:            
            #if kurosio(i, j) and marine_map[i][j]==marine_map[i][j]:
                if n<len(vec):
                    res[i][j] = vec[n]
                    n+=1
            else:
                res[i][j] = np.nan
    return res

def kurosio_vec_to_map_pooled(vec, marine_map_pooled):
    res = np.zeros(map_pooled_size)
    
    n = 0
    for i in range(map_pooled_size[0]):
        for j in range(map_pooled_size[1]):
            if kurosio_pooled(map_pooled_size[0]-i, j) and marine_map_pooled[i][j]==marine_map_pooled[i][j]:            
                if n<len(vec):
                    res[i][j] = vec[n]
                    n+=1
            else:
                res[i][j] = np.nan
    return res

# 黒潮マップ
kurosio_map = np.zeros((map_size_ais)) * np.nan
for i in range(map_size_ais[0]):
    for j in range(map_size_ais[1]):
        if kurosio(map_size_ais[0]-i, j):
            kurosio_map[i][j] = 1

kurosio_map_pooled = np.zeros((map_pooled_size)) * np.nan
for i in range(map_pooled_size[0]):
    for j in range(map_pooled_size[1]):
        if kurosio_pooled(map_pooled_size[0]-i, j):
            kurosio_map_pooled[i][j] = 1

kurosio_map_tf = np.zeros((map_size_ais)) == 1 
for i in range(map_size_ais[0]):
    for j in range(map_size_ais[1]):
        if kurosio(map_size_ais[0]-i, j):
            kurosio_map_tf[i][j] = True

kurosio_map_tf_pooled = np.zeros((map_pooled_size)) == 1 
for i in range(map_pooled_size[0]):
    for j in range(map_pooled_size[1]):
        if kurosio_pooled(map_pooled_size[0]-i, j):
            kurosio_map_tf_pooled[i][j] = True

kurosio_index = np.zeros((map_size_ais)) - 1 
count = 0
for i in range(map_size_ais[0]):
    for j in range(map_size_ais[1]):
        if kurosio(map_size_ais[0]-i, j) and nan_map[i][j]==nan_map[i][j]:
            kurosio_index[i][j] = int(count)
            count += 1

kurosio_index_pooled = np.zeros((map_pooled_size)) - 1 
count = 0
for i in range(map_pooled_size[0]):
    for j in range(map_pooled_size[1]):
        if kurosio_pooled(map_pooled_size[0]-i, j) and nan_map_pooled[i][j]==nan_map_pooled[i][j]:
            kurosio_index_pooled[i][j] = int(count)
            count += 1

kurosio_grid = []
for i in range(map_size_ais[0]):
    for j in range(map_size_ais[1]):
        if kurosio(map_size_ais[0]-i, j) and nan_map[i][j]==nan_map[i][j]:
            kurosio_grid.append([i, j])
kurosio_grid = np.array(kurosio_grid)

kurosio_grid_pooled = []
for i in range(map_pooled_size[0]):
    for j in range(map_pooled_size[1]):
        if kurosio_pooled(map_pooled_size[0]-i, j) and nan_map_pooled[i][j]==nan_map_pooled[i][j]:
            kurosio_grid_pooled.append([i, j])
kurosio_grid_pooled = np.array(kurosio_grid_pooled)



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

def dms_to_deg(dms):
    dms_value = float(dms)
    degrees = int(dms_value)
    minutes = int((dms_value - degrees) * 100)
    seconds = ((dms_value - degrees) * 100 - minutes) * 100
    decimale = degrees + (minutes / 60) + (seconds / 3600)
    return decimale/100

def latlon_to_mesh(lat, lon, deg_per_mesh=1/36, size=[1082, 1190], latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    grid0 = size[0] - int((lat-latlon_range[0])/deg_per_mesh)
    grid1 = int((lon-latlon_range[1])/deg_per_mesh)
    if grid0<0 or grid0>size[0]: return [-1, -1]
    if grid1<0 or grid1>size[1]: return [-1, -1]
    return [grid0, grid1]

def mesh_to_latlon(grid0, grid1, deg_per_mesh=1/36, size=[1082, 1190], latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    lat = (-grid0+size[0])*deg_per_mesh+latlon_range[0] 
    lon = grid1*deg_per_mesh+latlon_range[1] 
    return [lat, lon]

def latlon_to_mesh_df(lat, lon, deg_per_mesh=1/36, size=[1050, 1191], latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    grid0 = size[0] - ((lon-latlon_range[1]).astype(int)/deg_per_mesh)
    grid1 = ((lat-latlon_range[0]).astype(int)/deg_per_mesh)
    grid0[grid0 < 0] = -1
    grid0[grid0 > size[0]] = -1
    grid1[grid1 < 0] = -1
    grid1[grid1 > size[1]] = -1
    return grid0, grid1

def aisidx_to_rallon(lat_idx, lon_idx):
    return [lat_idx, lon_idx]*deg_per_mesh



def azimuth_to_radian(azimuth):
    radian = 360 - azimuth
    radian += 90
    radian = radian*np.pi/180
    return radian

def plot_target_points_mesh(grids, base_map=np.ones((1050, 1191))*nan_map[:1050], linewidth=1, half_size=50):
    # ex) grids = [[10, 20], [15, 25]]
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
    # 指定された時間数をdatetime.timedeltaオブジェクトとして作成
    delta = datetime.timedelta(hours=hours)
    # 指定された時間数分後の日時を計算
    target_dt = base_dt + delta
    return target_dt
    
# jcope
jcope_data = dummy_jcope_data(0, 0.1, 100, 0.01)

# ais(観測値) 
ais_data = dummy_ais_data(0, 0.1, 1, 100, 0.001)
