import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

num_data = 2
save_path = r"E:/shunsukeE/result/kalman-test/" 

def load_xzj(day, hour, num_data, thres=0.6, save_path=''):
    pathX = save_path + fr'saverX201509{day:02}{hour:02}-N{num_data}.pkl'
    pathXCur = save_path + fr'saverXCur201509{day:02}{hour:02}-N{num_data}.pkl'
    pathZ = save_path + fr'saverZ201509{day:02}{hour:02}-N{num_data}.pkl'
    pathH = save_path + fr'saverH201509{day:02}{hour:02}-N{num_data}.pkl'
    pathP = save_path + fr'saverP201509{day:02}{hour:02}-N{num_data}.pkl'
    pathF = save_path + fr'saverF201509{day:02}{hour:02}-N{num_data}.pkl'
    pathR = save_path + fr'saverR201509{day:02}{hour:02}-N{num_data}.pkl'
    pathK = save_path + fr'saverK201509{day:02}{hour:02}-N{num_data}.pkl'
    pathJCOPE = save_path + fr'saverJCOPE201509{day:02}{hour:02}-N{num_data}.pkl'
    pathJCOPECur = save_path + fr'saverJCOPECur201509{day:02}{hour:02}-N{num_data}.pkl'

    x = pkl.load(open(pathX, 'rb'))
    jcope = pkl.load(open(pathJCOPE, 'rb'))

    xCur = pkl.load(open(pathXCur, 'rb'))
    jcopeCur = pkl.load(open(pathJCOPECur, 'rb'))
    z = pkl.load(open(pathZ, 'rb'))

    H = pkl.load(open(pathH, 'rb'))
    P = pkl.load(open(pathP, 'rb'))
    F = pkl.load(open(pathF, 'rb'))
    R = pkl.load(open(pathR, 'rb'))
    K = pkl.load(open(pathK, 'rb'))
    return x, jcope, xCur, jcopeCur, z, H, P, F, R, K  

def diff_xzj(x, z, j):
    thres = 0.6

    a = x-z
    print(f'kalman-ais')
    print(f'Mean: {np.mean(a)}')
    print(f'Under {thres} rate:{np.sum(a<thres)/np.sum(a==a)}')

    a = x-j
    print(f'kalman-jcope')
    print(f'Mean: {np.mean(a)}')
    print(f'Under {thres} rate:{np.sum(a<thres)/np.sum(a==a)}')

    a = z-j
    print(f'ais-jcope')
    print(f'Mean: {np.mean(a)}')
    print(f'Under {thres} rate:{np.sum(a<thres)/np.sum(a==a)}')
    

