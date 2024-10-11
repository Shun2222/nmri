import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from ais3_2detail import *
import os 
import os.path as osp
import math
import itertools
from scipy.spatial import distance_matrix
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
from tqdm import tqdm

def insecs_plot(data_info, ss, ss2, cur, line_plot=True, save=False, savepath='./'):
    lines = ss.split('\n')
    datas = []
    for i in range(len(lines)):
        if not 'elem' in lines[i]:
            continue
        line = lines[i].split(', ')
        dict = {}
        for j in range(len(line)):
            if j==0: continue
            data = line[j].split(':')
            dict[data[0]] = float(data[1])
        datas.append(dict)

    keys = datas[0].keys()
    print(keys)

    line_datas = {}
    for key in keys:
        line_datas[key] = []

    for d in datas:
        for key in keys:
            line_datas[key].append(d[key])
    for key in keys:
        line_datas[key] = np.array(line_datas[key])

    hdg_set = [] 
    for hdg in sorted(set(line_datas["Hdg"])):
        n = np.sum(line_datas['Hdg']==hdg)
        if n>2:
            hdg_set.append(hdg) 
    insec_type = {}
    n = 0
    for i in range(len(hdg_set)):
        for j in range(i+1, len(hdg_set)):
           insec_type[f"{hdg_set[i]}-{hdg_set[j]}"] = n
           n+=1
    max_insec_type = n-1

    def get_insec_type(line0, line1):
        hdg0 = line_datas["Hdg"][line0]
        hdg1 = line_datas["Hdg"][line1]
        if hdg0>hdg1:
            tmp = hdg0
            hdg0 = hdg1
            hdg1 = tmp

        keys = insec_type.keys()
        if hdg0==hdg1:
            return -1
        elif not f"{hdg0}-{hdg1}" in keys:
            return -1
        else:
            return insec_type[f"{hdg0}-{hdg1}"]


    lines = ss2.split('\n')
    datas = []
    for i in range(len(lines)):
        if not 'insec' in lines[i]:
            continue
        line = lines[i].split(', ')
        dict = {}
        is_avail = True
        for j in range(len(line)):
            if j==0:
                data = line[j].split('-')
                dict["insec_type"] = get_insec_type(int(data[0]), int(data[1]))
            else:    
                data = line[j].split(':')
                dict[data[0]] = float(data[1])
                if float(data[1])==999:
                    is_avail = False
        if is_avail:
            datas.append(dict)

    keys = datas[0].keys()
    print(keys)

    insec_datas = {}
    for key in keys:
        insec_datas[key] = []

    for d in datas:
        for key in keys:
            insec_datas[key].append(d[key])
    for key in keys:
        insec_datas[key] = np.array(insec_datas[key])

    insecXY = np.concatenate([np.array([insec_datas["insecX"]]), np.array([insec_datas["insecY"]])], axis=0).T
    target_insec_set = np.arange(max_insec_type)

    #print(f'hdgs: {hdg_set}')
    #print(f'insec type: {insec_datas["insec_type"]}')
    #print(f'insec set: {insec_set}')

    def create_distance_matrix(points):
        return distance_matrix(points, points)

    # グラフを作成してTSP問題を解く
    def solve_tsp(points):
        distance_matrix = create_distance_matrix(points)
        G = nx.Graph()
        num_points = distance_matrix.shape[0]
        
        # グラフにノードとエッジを追加
        for i in range(num_points):
            G.add_node(i)
            for j in range(i + 1, num_points):
                G.add_edge(i, j, weight=distance_matrix[i, j])
        
        # TSP問題を解く
        tsp_path = traveling_salesman_problem(G, cycle=False, weight='weight')
        
        # 最短経路の長さを計算
        length = sum(G[u][v]['weight'] for u, v in zip(tsp_path, tsp_path[1:] + [tsp_path[0]]))
        return length
    """  面積を求めるアルゴリズム
    def polygon_area(points):
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[i][1] * points[j][0]
        area = abs(area) / 2.0
        return area
    """
    min_area = -1
    min_comb = None
    insec_type_num = len(target_insec_set)

    if insec_type_num>1:
        x = insec_datas['numX']
        y = insec_datas['numY']
        num = np.abs(x) + np.abs(y)
        tf = np.min(num)==num
        min_num_insec = insecXY[tf]
        min_num_insec_type = insec_datas['insec_type'][tf]
        insec_sets = []
        for insec, insec_type in zip(min_num_insec, min_num_insec_type):
            insec_set = [np.array([insec])]
            for i in target_insec_set:
                if i==insec_type:
                    continue
                insec_set.append(insecXY[insec_datas["insec_type"]==i])
            insec_sets.append(insec_set)
        for insec_set in tqdm(insec_sets):
            combinations = list(itertools.product(*insec_set))
            print(np.array(combinations).shape)
            print(f'\rcomputing tsp now...', end='')
            for comb in combinations:
                comb = np.array(comb) 
                area = solve_tsp(comb)
                # area = polygon_area(comb)
                if min_area==-1:
                    min_area = area
                    min_comb = comb
                elif min_area>area:
                    min_area = area
                    min_comb = comb
        print(f'\rfinished computing tsp.', end='')
    cur_value = cur[0].split(', ')
    cur_value = [float(c.split(':')[1]) for c in cur_value]
    cur_value2 = cur[1].split(', ')
    cur_value2 = [float(c.split(':')[1]) for c in cur_value2]

    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    for ax in [ax0, ax1]:
        a = line_datas['tanHdg']
        x = line_datas['vogN']
        y = line_datas['vogE']
        b = y - a*x 
        x = np.arange(-5, 5, 0.1)
        y = [x*a[i]+b[i] for i in range(len(a))]
        line_type = list(set(a))

        cmap = cm.get_cmap("hsv")
        color_value = line_datas['Hdg'] - np.min(line_datas['Hdg']) + 1
        color_value[color_value>=180] -= 180
        colors = cmap( color_value / np.max(color_value))
        if ax==ax1:
            for i in range(len(y)):
                idx = np.argmax(a[i]==line_type)
                idx = idx if idx<len(colors) else len(colors)-1
                #plt.plot(x, y[i], color=colors[idx])
                ax.plot(x, y[i], color=colors[i])

        cmap = cm.get_cmap("hsv")
        color_value = insec_datas['insec_type']+1
        colors = cmap( color_value / np.max(color_value))
        ax.scatter(insec_datas['insecX'], insec_datas['insecY'], color=colors, alpha=0.3)
        ax.scatter(cur_value[0], cur_value[1], color='black', alpha=1.0, marker="*")
        ax.scatter(cur_value2[0], cur_value2[1], color='orange', alpha=1.0, marker="*")

        if insec_type_num>1:
            ax.fill(min_comb[:, 0], min_comb[:, 1], 'c', edgecolor='k', alpha=0.5)
            ax.scatter(np.mean(min_comb[:, 0]), np.mean(min_comb[:, 1]), color='r', alpha=1.0, marker="*")

        for i in range(len(insec_datas['insecX'])):
            x = insec_datas['numX'][i]
            y = insec_datas['numY'][i]
            t = f"{x}, {y}"
            #plt.text(insec_datas['insecX'][i], insec_datas['insecY'][i], t)


        ax.set_xlabel('N')
        ax.set_ylabel('E')
    plt.title(data_info)
    if save:
        os.makedirs(savepath, exist_ok=True)
        name = osp.join(savepath, data_info+'.png')
        plt.savefig(name)
        print(f'saved in {name}')
    else:
        plt.show()
    plt.close()

    #for key in line_datas.keys():
    #    print(f'{key}: {set(line_datas[key])}')

if __name__ == "__main__":
    #data_info, ss, ss2, cur = get_ais3_2detail()
    data_infos, sss, ss2s, curs = get_all_ais3_2detail(max_data_num=10)
    for data_info, ss, ss2, cur in zip(data_infos, sss, ss2s, curs):
        insecs_plot(data_info, ss, ss2, cur, line_plot=False, save=True, savepath='images/test5')
    print(f'finished saving insecs plot.\n')
