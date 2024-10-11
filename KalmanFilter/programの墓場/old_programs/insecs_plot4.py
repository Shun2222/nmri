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


    mmsi_set = sorted(set(line_datas["mmsi"]))

    if not 441378000 in mmsi_set:
        print(f'Not exit target mmsi')
        #return

    mmsi_type = {}
    n = 0
    for i in mmsi_set:
        mmsi_type[int(i)] = n
        n+=1
    max_mmsi_type = n-1

    def get_insec_key(line0, line1):
        hdg0 = line_datas["Hdg"][line0]
        hdg1 = line_datas["Hdg"][line1]
        if hdg0>hdg1:
            tmp = hdg0
            hdg0 = hdg1
            hdg1 = tmp
        return f"{hdg0}-{hdg1}"

    lines = ss2.split('\n')
    datas = []
    insec_tf = {}
    for i in range(len(lines)):
        if not 'insec' in lines[i]:
            continue
        line = lines[i].split(', ')
        dict = {}
        is_avail = True
        for j in range(len(line)):
            if j==0:
                data = line[j].split('-')
                dict["elem"] = [int(data[0]), int(data[1])]
            else:    
                data = line[j].split(':')
                dict[data[0]] = float(data[1])
                if float(data[1])==999:
                    is_avail = False
        if is_avail:
            datas.append(dict)
            insec_tf[get_insec_key(dict['elem'][0], dict['elem'][1])] = True

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

    hdg_set = [] 
    line_count_thres=1
    for hdg in sorted(set(line_datas["Hdg"])):
        n = np.sum(line_datas['Hdg']==hdg)
        if n>line_count_thres:
            hdg_set.append(hdg) 
    insec_type = {}
    n = 0
    for i in range(len(hdg_set)):
        for j in range(i+1, len(hdg_set)):
            if f"{hdg_set[i]}-{hdg_set[j]}" in insec_tf.keys():
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

    insec_datas["insec_type"] = np.array([get_insec_type(l0, l1) for l0, l1 in insec_datas['elem']])

    insecXY = np.concatenate([np.array([insec_datas["insecX"]]), np.array([insec_datas["insecY"]])], axis=0).T
    target_insec_set = np.arange(max_insec_type+1)

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

    if insec_type_num>=1:
        x = insec_datas['numX']
        y = insec_datas['numY']
        num = np.abs(x) + np.abs(y)
        tf = np.min(num)==num
        min_num_insec = insecXY[tf]
        min_num_insec_type = insec_datas['insec_type'][tf]
        insec_sets = []
        for insec, insec_type in zip(min_num_insec, min_num_insec_type):
            dist = np.zeros(insec_type_num)
            comb = [insec]
            for i in target_insec_set:
                if i==insec_type:
                    continue
                d = np.sum((insec-insecXY[insec_datas["insec_type"]==i])**2, axis=1)
                dist[i] = np.min(d)
                comb.append(insecXY[insec_datas["insec_type"]==i][np.argmin(d)])
            if min_area==-1:
                min_area = np.sum(dist) 
                min_comb = np.array(comb)
            elif min_area>np.sum(dist):
                min_area = np.sum(dist) 
                min_comb = np.array(comb)

        print(f'\rfinished computing tsp.', end='')

    print(cur[0])
    cur_value = cur[0][0].split(', ')
    cur_value = [float(c.split(':')[1]) for c in cur_value]
    cur_value2 = cur[0][1].split(', ')
    cur_value2 = [float(c.split(':')[1]) for c in cur_value2]
    cur_value3 = cur[0][2].split(', ')
    cur_value3 = [float(c.split(':')[1]) for c in cur_value3]

    cur_value4 = []
    if not '∞' in cur[0][3]:
        cur_value4 = cur[0][3].split(', ')
        cur_value4 = [float(c.split(':')[1]) for c in cur_value4]

    cur_mmsi_datas = []
    for cur_mmsi in cur[1]:
        data = []
        cur_mmsi = cur_mmsi.split(', ')
        isMmsi = True
        for cmm in cur_mmsi:
            if "curN" in cmm:
                isMmsi = False
            if isMmsi:
                if ":" in cmm:
                    mmsi = int(cmm.split(':')[1])
                else:
                    mmsi = int(cmm)
                data.append(mmsi) 
            else:
                data.append(float(cmm.split(':')[1]))

        cur_mmsi_datas.append(data)
    bad_mmsi = []
    if cur[2][0]=='':
        mmsi = cur[2][0]
        mmsi = mmsi.split(', ')
        for cmm in mmsi[:-1]:
            if ":" in cmm:
                mmsi = int(cmm.split(':')[1])
            else:
                mmsi = int(cmm)
            bad_mmsi.append(mmsi)

    fig = plt.figure(figsize=(32, 16))
    ax0 = fig.add_subplot(2, 4, 1)
    ax01 = fig.add_subplot(2, 4, 2)
    ax02 = fig.add_subplot(2, 4, 3)
    ax1 = fig.add_subplot(2, 4, 5)
    ax2 = fig.add_subplot(2, 4, 6)
    for ax in [ax0, ax01, ax02, ax1, ax2]:
        a = line_datas['tanHdg']
        x = line_datas['vogN']
        y = line_datas['vogE']
        b = y - a*x 
        x = np.arange(-5, 5, 0.1)
        y = [x*a[i]+b[i] for i in range(len(a))]
        line_type = list(set(a))

        if ax==ax1:
            cmap = cm.get_cmap("hsv")
            color_value = line_datas['Hdg'] - np.min(line_datas['Hdg']) + 1
            color_value[color_value>=180] -= 180
            colors = cmap( color_value / np.max(color_value))
            for i in range(len(y)):
                idx = np.argmax(a[i]==line_type)
                idx = idx if idx<len(colors) else len(colors)-1
                #plt.plot(x, y[i], color=colors[idx])
                ax.plot(x, y[i], color=colors[i])
        if ax==ax2:
            cmap = cm.get_cmap("hsv")
            color_value = np.array([mmsi_type[int(mmsi)] for mmsi in line_datas['mmsi']]) + 1
            colors = cmap( color_value / np.max(color_value))
            ploted_mmsi = []
            for i in range(len(y)):
                idx = np.argmax(a[i]==line_type)
                idx = idx if idx<len(colors) else len(colors)-1
                #plt.plot(x, y[i], color=colors[idx])
                if line_datas['mmsi'][i] in ploted_mmsi:
                    ax.plot(x, y[i], color=colors[i])
                else:
                    ploted_mmsi.append(line_datas['mmsi'][i])
                    ax.plot(x, y[i], color=colors[i], label=int(line_datas['mmsi'][i]))

        cmap = cm.get_cmap("hsv")
        color_value = insec_datas['insec_type']+1
        colors = cmap( color_value / np.max(color_value))
        if ax!=ax01 and ax!=ax02:
            ax.scatter(insec_datas['insecX'], insec_datas['insecY'], color=colors, alpha=0.3)
        """
        if np.abs(cur_value[0]) + np.abs(cur_value[1]) < 10:
            ax.scatter(cur_value[0], cur_value[1], color='black', alpha=1.0, marker="*", label="InsecsLSM")
        if np.abs(cur_value3[0]) + np.abs(cur_value3[1]) < 10:
            ax.scatter(cur_value3[0], cur_value3[1], color='green', alpha=1.0, marker="*", label="HdgLSM")
        if insec_type_num>=1:
            import matplotlib.collections as mc
            lines = [[min_comb[0], min_comb[i+1]] for i in range(len(min_comb)-1)]
            lc = mc.LineCollection(lines, color='black', linewidths=0.5)
            ax.add_collection(lc)
            ax.scatter(np.mean(min_comb[:, 0]), np.mean(min_comb[:, 1]), color='r', alpha=1.0, marker="*", label="DiffTypeInsecsMinDist")

        for i in range(len(insec_datas['insecX'])):
            x = insec_datas['numX'][i]
            y = insec_datas['numY'][i]
            t = f"{x}, {y}"
            #plt.text(insec_datas['insecX'][i], insec_datas['insecY'][i], t)
        """

        if len(cur_mmsi_datas)>0:
            cmap = cm.get_cmap("hsv")
            color_value = len(cur_mmsi_datas)
            color_value = np.arange(color_value) + 1
            colors = cmap( color_value / np.max(color_value))
            is_first = [False, False]
            for c, cur_mmsi in zip(colors, cur_mmsi_datas):
                mmsi_str = "MMSI:"
                is_enclude_bad_mmsi = False
                for i in range(len(cur_mmsi)-2):
                    mmsi_str += str(cur_mmsi[i]) + ", "
                    if cur_mmsi[i] in bad_mmsi:
                        is_enclude_bad_mmsi = True
                mmsi_str =  mmsi_str[0:-2]
                if ax==ax0:
                    ax.scatter(cur_mmsi[-2], cur_mmsi[-1], s=50, marker='*', color=c, alpha=1.0, label=mmsi_str)
                if ax==ax02:
                    ax.scatter(cur_mmsi[-2], cur_mmsi[-1], s=50, color=c, alpha=1.0, label=mmsi_str)
                elif ax==ax01:
                    if is_enclude_bad_mmsi:
                        if not is_first[0]:
                            ax.scatter(cur_mmsi[-2], cur_mmsi[-1], s=50, color="gray", alpha=1.0, label='enclude bad mmsi')
                            is_first[0] = True
                        ax.scatter(cur_mmsi[-2], cur_mmsi[-1], s=50, color="gray", alpha=1.0)
                    else:
                        if len(cur_mmsi)-2==len(mmsi_set)-len(bad_mmsi):
                            ax.scatter(cur_mmsi[-2], cur_mmsi[-1], s=80, marker=',', color=c, alpha=1.0, label='LSM without badMmsi in grid')
                        else:
                            ax.scatter(cur_mmsi[-2], cur_mmsi[-1], s=50, color=c, alpha=1.0)

        if len(cur_value4)!=0:
            if np.abs(cur_value4[0]) + np.abs(cur_value4[1]) < 10:
                if ax==ax01:
                    ax.scatter(cur_value4[0], cur_value4[1], s=80, marker="*", color='blue', alpha=1.0, label="Conventional LSM")
                else:
                    ax.scatter(cur_value4[0], cur_value4[1], s=80, marker="*", color='blue', alpha=1.0)

        #if np.abs(cur_value[0]) + np.abs(cur_value[1]) < 10:
        if True:
            if ax==ax01:
                ax.scatter(cur_value[0], cur_value[1], s=80, color='red', alpha=1.0, marker="*", label="LSM without allBadMmsi")
            else:
                ax.scatter(cur_value[0], cur_value[1], s=80, color='red', alpha=1.0, marker="*")

        ax.grid(True)
        ax.set_xlabel('N')
        ax.set_ylabel('E')
    if save:
        added_black = "addedBlackMmsi:" + cur[3][0] if len(cur[3])==1 else ""
        ax0.set_title(data_info+", "+cur[2][0]+", "+cur[2][1] + ", " + added_black)
        ax01.legend()
        ax02.legend(loc='upper center', bbox_to_anchor=(1.5, 1.05))
        ax2.legend()

        os.makedirs(savepath, exist_ok=True)
        target_mmsi_savepath = osp.join(savepath, "ExistLambdaLSM")
        os.makedirs(target_mmsi_savepath, exist_ok=True)
        not_target_mmsi_savepath = osp.join(savepath, "else")
        os.makedirs(not_target_mmsi_savepath, exist_ok=True)

        name = osp.join(not_target_mmsi_savepath, data_info+".png")
        if len(cur_value4)!=0:
            if np.abs(cur_value4[0]) + np.abs(cur_value4[1]) < 10:
                name = osp.join(target_mmsi_savepath, data_info+".png")
        plt.savefig(name)
        print(f'saved in {name}')
    else:
        plt.show()
    plt.close()

    #for key in line_datas.keys():
    #    print(f'{key}: {set(line_datas[key])}')

if __name__ == "__main__":
    #data_info, ss, ss2, cur = get_ais3_2detail()
    data_infos, sss, ss2s, curs = get_all_ais3_2detail(max_data_num=1e3)
    for data_info, ss, ss2, cur in zip(data_infos, sss, ss2s, curs):
        insecs_plot(data_info, ss, ss2, cur, line_plot=False, save=True, savepath='images/test/test11')
    print(f'finished saving insecs plot.\n')
