import numpy as np
from utils import *
from utils.kalman_parameters import *
from utils.utils_needed_params import *

_2N0 = len(kurosio_grid_pooled) 
print(_2N0)
print(kurosio_index_pooled.shape)

def B_mat2():
    B = np.zeros((_2N0+1, _2N0+1))
    range_lat = 5 # TODO
    range_lon = 10 # TODO
    lat_halflife = 0.037
    lon_halflife = 0.064
    deg_per_mesh = 1/(36/pool_size)
    theta = -np.arctan(1/3)
    for i in range(_2N0):
        history = []
        grid0, grid1 = kurosio_grid_pooled[i]
        for dy in np.arange(-10, 10):
            for dx in np.arange(-20, 20):
                # 重みの半減距離は度数で計算するため、度数に変換
                dydeg = deg_per_mesh*(dy+dx)/(2*np.cos(theta))
                dxdeg = deg_per_mesh*(dy-dx)/(2*np.sin(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))

                if not grid0+dy>=kurosio_index_pooled.shape[0] and not grid1+dx>=kurosio_index_pooled.shape[1]:
                    idx = int(kurosio_index_pooled[grid0+dy][grid1+dx])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w
    for i in range(_2N0):
        B[i][i] = 1.0
    return B

def B_mat3():
    B = np.zeros((_2N0+1, _2N0+1))
    range_lat = 5 # TODO
    range_lon = 10 # TODO
    lat_halflife = 0.037
    lon_halflife = 0.064
    pool_size = 2
    deg_per_mesh = 1/(36/pool_size)
    theta = -np.arctan(1/3)
    #for i in range(_2N0):
    for i in range(2):
        i = 300
        history = []
        grid0, grid1 = kurosio_grid_pooled[i]
        for dy in np.arange(0, range_lat, 0.3):
            for dx in np.arange(0, range_lon, 0.3):

                # 黒潮方向dy2とdy2+90°方向dx2を計算
                dy2 = int(dy*np.cos(theta) + dx*np.sin(theta))
                dx2 = int(dx*np.cos(theta) - dy*np.sin(theta))
                dydeg = deg_per_mesh*(dy2+dx2)/(2*np.cos(theta))
                dxdeg = deg_per_mesh*(dy2-dx2)/(2*np.sin(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w
                        print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                dy2 = int(dy*np.cos(theta) - dx*np.sin(theta))
                dx2 = int(-dx*np.cos(theta) - dy*np.sin(theta))
                dydeg = deg_per_mesh*(dy2+dx2)/(2*np.cos(theta))
                dxdeg = deg_per_mesh*(dy2-dx2)/(2*np.sin(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w
                        print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                dy2 = int(-dy*np.cos(theta) + dx*np.sin(theta))
                dx2 = int(dx*np.cos(theta) + dy*np.sin(theta))
                dydeg = deg_per_mesh*(dy2+dx2)/(2*np.cos(theta))
                dxdeg = deg_per_mesh*(dy2-dx2)/(2*np.sin(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w
                        print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                dy2 = int(-dy*np.cos(theta) - dx*np.sin(theta))
                dx2 = int(-dx*np.cos(theta) + dy*np.sin(theta))
                dydeg = deg_per_mesh*(dy2+dx2)/(2*np.cos(theta))
                dxdeg = deg_per_mesh*(dy2-dx2)/(2*np.sin(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w
                        print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')



    for i in range(_2N0):
        B[i][i] = 1.0
    return B


def B_mat():
    B = np.zeros((_2N0+1, _2N0+1))
    range_lat = 5 # TODO
    range_lon = 10 # TODO
    lat_halflife = 0.037
    lon_halflife = 0.064
    pool_size = 2
    deg_per_mesh = 1/(36/pool_size)
    theta = -np.arctan(1/3)
    for i in range(2):
        i = 300
        history = []
        grid0, grid1 = kurosio_grid_pooled[i]
        for dy in np.arange(0, range_lat, 0.3):
            for dx in np.arange(0, range_lon, 0.3):
                # 重みの半減距離は度数で計算するため、度数に変換
                dydeg =  dy*deg_per_mesh # dy2方向の度数 
                dxdeg =  dx*deg_per_mesh # dx2方向の度数
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))

                # 黒潮方向dy2とdy2+90°方向dx2を計算
                dy2 = int(dy*np.cos(theta) + dx*np.sin(theta))
                dx2 = int(dx*np.cos(theta) - dy*np.sin(theta))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                    if idx!=-1: 
                        B[i][idx] = w
                dy2 = int(dy*np.cos(theta) - dx*np.sin(theta))
                dx2 = int(-dx*np.cos(theta) - dy*np.sin(theta))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w

                dy2 = int(-dy*np.cos(theta) + dx*np.sin(theta))
                dx2 = int(dx*np.cos(theta) + dy*np.sin(theta))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w

                dy2 = int(-dy*np.cos(theta) - dx*np.sin(theta))
                dx2 = int(-dx*np.cos(theta) + dy*np.sin(theta))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i][idx] = w

    for i in range(_2N0):
        B[i][i] = 1.0
    return B

import scipy.sparse as sps
def B_mat_sp():
    B = sps.lil_matrix(np.zeros((_2N0+1, _2N0+1), dtype=np.float32))
    range_lat = 5 # TODO
    range_lon = 10 # TODO
    lat_halflife = 0.037
    lon_halflife = 0.064
    pool_size = 1
    deg_per_mesh = 1/(36/pool_size)
    theta = -np.arctan(1/3)
    _N0 = int(_2N0/2)
    for i in range(2):
        i = 300 
        grid0, grid1 = kurosio_grid_pooled[i]
        for dy in np.arange(0, range_lat, 0.3):
            for dx in np.arange(0, range_lon, 0.3):
                # 重みの半減距離は度数で計算するため、度数に変換
                dydeg =  dy*deg_per_mesh # dy2方向の度数 
                dxdeg =  dx*deg_per_mesh # dx2方向の度数
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))

                # 以下、４つの方向でそれぞれ重みを計算し、行列Bに反映
                # 黒潮方向dy2とdy2+90°方向dx2を計算
                dy2 = int(dy*np.cos(theta) + dx*np.sin(theta))
                dx2 = int(dx*np.cos(theta) - dy*np.sin(theta))
                if [dy2, dx2] in [[1,0], [1,1], [2,1]]:
                    lat = dy2 * np.cos(theta) - dx2*np.sin(theta)
                    lon = dy2 * np.sin(theta) + dx2*np.cos(theta)
                    dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                    dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                    w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                    print(f'dy,dx,lat,lon,dydeg,dxdeg, {dy2}, {dx2}, {lat}, {lon}, {dydeg}, {dxdeg}, {w}')
                    input()
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w
                        if w>0.3:
                            print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                dy2 = int(dy*np.cos(theta) - dx*np.sin(theta))
                dx2 = int(-dx*np.cos(theta) - dy*np.sin(theta))
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w
                        if w>0.3:
                            print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                dy2 = int(-dy*np.cos(theta) + dx*np.sin(theta))
                dx2 = int(dx*np.cos(theta) + dy*np.sin(theta))
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w
                        if w>0.3:
                            print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')

                dy2 = int(-dy*np.cos(theta) - dx*np.sin(theta))
                dx2 = int(-dx*np.cos(theta) + dy*np.sin(theta))
                dydeg = deg_per_mesh*np.abs(dy2 * np.cos(theta) - dx2*np.sin(theta))
                dxdeg = deg_per_mesh*np.abs(dy2 * np.sin(theta) + dx2*np.cos(theta))
                w = ((1/2) ** (dydeg/lat_halflife)) * ((1/2) ** (dxdeg/lon_halflife))
                if not grid0+dy2>=kurosio_index_pooled.shape[0] and not grid1+dx2>=kurosio_index_pooled.shape[1]\
                and not grid0<dy2 and not grid1<dx2:
                    idx = int(kurosio_index_pooled[grid0+dy2][grid1+dx2])
                    # print(f'dx2,dy2 = {dx2},{dy2} (idx={idx})')
                    if idx!=-1: 
                        B[i, idx] = w
                        B[i+_N0, idx] = w
                        if w>0.3:
                            print(f'dx2,dy2,w = {dx2},{dy2},{w} (idx={idx})')
    return B.toarray()

def test(save=True):
    target_idx = 300 
    B = B_mat_sp()
    B_map = kurosio_vec_to_map_pooled(B[target_idx], nan_map_pooled) * nan_map_pooled
    if save:
        plt.figure(figsize=(24, 24))
        plt.title(f'B {target_idx}')
        sns.heatmap(B_map, cbar=True)
        path = f'logs/B{target_idx}.png'
        plt.savefig(path)
        plt.close()

        B_mapArea = B_map[290:310].T[310:330].T
        plt.figure(figsize=(24, 24))
        plt.title(f'B {target_idx}')
        sns.heatmap(B_mapArea, cbar=True, annot=True, fmt='1.3f')
        path = f'logs/B{target_idx}-up.png'
        plt.savefig(path)
        plt.savefig(path)
        plt.close()
    B_mapArea = B_map[270:330].T[270:330].T
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x = np.arange(len(B_mapArea[0]))
    y = np.arange(len(B_mapArea[1]))
    x, y = np.meshgrid(x, y)
    print(x.shape)
    print(y.shape)
    print(B_map.shape)
    print(B_mapArea.shape)
    ax.plot_surface(x, y, B_mapArea, cmap='summer')
    plt.show()

if __name__ =="__main__":
    test()
