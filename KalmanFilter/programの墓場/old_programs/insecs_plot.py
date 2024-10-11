import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from ais3_2detail import *
import os 
import os.path as osp


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


    lines = ss2.split('\n')
    datas = []
    for i in range(len(lines)):
        if not 'insec' in lines[i]:
            continue
        line = lines[i].split(', ')
        dict = {}
        is_avail = True
        for j in range(len(line)):
            if j==0: continue
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

        cmap = cm.get_cmap('hsv')
        color_value = line_datas['Hdg'] - np.min(line_datas['Hdg']) + 1
        color_value[color_value>=180] -= 180
        colors = cmap( color_value / np.max(color_value))
        if ax==ax1:
            for i in range(len(y)):
                idx = np.argmax(a[i]==line_type)
                idx = idx if idx<len(colors) else len(colors)-1
                #plt.plot(x, y[i], color=colors[idx])
                ax.plot(x, y[i], color=colors[i])

        sumNum = (np.abs(insec_datas['numX']) + np.abs(insec_datas['numY']))
        ax.scatter(insec_datas['insecX'][sumNum>3], insec_datas['insecY'][sumNum>3], color='b')
        ax.scatter(insec_datas['insecX'][sumNum<=3], insec_datas['insecY'][sumNum<=3], color='g')
        tfs = np.min(sumNum)==sumNum
        ax.scatter(insec_datas['insecX'][tfs], insec_datas['insecY'][tfs], color='orange')
        insec_x = insec_datas['insecX'][tfs].reshape(np.sum(tfs), 1)
        insec_y = insec_datas['insecY'][tfs].reshape(np.sum(tfs), 1)
        insec_xy = np.concatenate([insec_x, insec_y], axis=1)
        print(f'xy(min sumNum):\n {insec_xy}\n')

        for i in range(len(insec_datas['insecX'])):
            x = insec_datas['numX'][i]
            y = insec_datas['numY'][i]
            t = f"{x}, {y}"
            #plt.text(insec_datas['insecX'][i], insec_datas['insecY'][i], t)

        ax.scatter(cur_value[0], cur_value[1], color='black')
        ax.scatter(cur_value2[0], cur_value2[1], color='r')

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
    data_infos, sss, ss2s, curs = get_all_ais3_2detail(max_data_num=1000)
    for data_info, ss, ss2, cur in zip(data_infos, sss, ss2s, curs):
        insecs_plot(data_info, ss, ss2, cur, line_plot=False, save=True, savepath='images/test')
    print(f'finished saving insecs plot.\n')
