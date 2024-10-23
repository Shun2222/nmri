import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import math
import pickle as pkl
import os.path as osp

from .utils import *

project = get_project_name()
if 'kalman' in project:
    from .kalman_parameters import *
elif 'analysis' in project:
    from .analysis_parameters import *
elif 'Ais4' in project:
    from .kalman_parameters import *
elif 'shipLog' in project:
    from .shipLog_parameters import *
elif 'kernel' in project:
    from .analysis_parameters import *
else:
    #from .kalman_parameters import *
    from .analysis_parameters import *
    
    

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
            if not np.all(np.isnan(image[y_start:y_end].T[x_start:x_end])):
                pooled[y, x] = np.nanmean(image[y_start:y_end].T[x_start:x_end])
            else:
                pooled[y, x] = np.nan
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
    assert type(data)==np.ndarray, f"type={type(data)}"
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

kurosio_index = np.zeros((map_size_ais), dtype=np.int64) - 1 
count = 0
for i in range(map_size_ais[0]):
    for j in range(map_size_ais[1]):
        if kurosio(map_size_ais[0]-i, j) and nan_map[i][j]==nan_map[i][j]:
            kurosio_index[i][j] = int(count)
            count += 1

kurosio_index_pooled = np.zeros((map_pooled_size), dtype=np.int64) - 1 
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
kurosio_grid = np.array(kurosio_grid, dtype=np.int64)

kurosio_grid_pooled = []
for i in range(map_pooled_size[0]):
    for j in range(map_pooled_size[1]):
        if kurosio_pooled(map_pooled_size[0]-i, j) and nan_map_pooled[i][j]==nan_map_pooled[i][j]:
            kurosio_grid_pooled.append([i, j])
kurosio_grid_pooled = np.array(kurosio_grid_pooled)