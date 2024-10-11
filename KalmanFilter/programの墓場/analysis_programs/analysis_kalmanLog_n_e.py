import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from utils import * 

year = 2015
month = 9
n_day = 15 #nday_month(month)
n_hour = 12 #24
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)

print(f'Loading data')
al = AISLoader(2015, 9)
ais_keys = ['n', 'e', 'd']
al.load_path(keys=ais_keys)

kl = KalmanLogLoader(2015, 9, 20)
log_path = r"E:\shunsukeE\result\kalman-BQBtest-Q001"

path_ais = r'E:\shunsukeE\data\ais\1509-ais4s-pkls'
kalman_keys = ['X', 'Z', 'Target']
kl.set_path(log_path)

jl = JCOPELoader(2015, 9)
path_jcope = fr'E:\shunsukeE\data\eas2'
jl.load_path(path_jcope)

n_data = -1
for s in range(0, 1):
    ais_n = []
    ais_e = []
    ais_d = []
    kalman_n = []
    kalman_e = []
    jcope_n = []
    jcope_e = []
    for day in range(1, n_day+1):
        print(f's:{s}, day:{day}')
        #TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']
        start_hour = 0 if day == 1 else 1

        data = kl.load_kalmanLog_day(day, s, keys=kalman_keys)
        for hour in range(start_hour, n_hour):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue

            if n_data == -1:
                n_data = int((data['X'][1].shape[0] - 1)/2)
                print(n_data)
                TF = np.array([True]*n_data)

            kalman_n.append(data['X'][dtidx][:np.sum(TF)])
            kalman_e.append(data['X'][dtidx][np.sum(TF):-1])

        for hour in range(start_hour, n_hour):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue

            path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}X.csv"
            dataN = pd.read_csv(path, encoding="cp932", header=None)
            dataN = dataN.values
            ais_n.append(kurosio_filter_pooled(dataN, nan_map_pooled, is_pooled=True)[TF])

            path = f"{path_ais}/AisCurr{year}{month:02}{day:02}{hour:02}Y.csv"
            dataE = pd.read_csv(path, encoding="cp932", header=None)
            dataE = dataE.values
            ais_e.append(kurosio_filter_pooled(dataE, nan_map_pooled, is_pooled=True)[TF])

        data_n, data_e = jl.load_jcope_day(day)
        for hour in range(start_hour, n_hour):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue
            jcope_n.append(kurosio_filter_pooled(data_n[dtidx], nan_map_pooled)[TF])
            jcope_e.append(kurosio_filter_pooled(data_e[dtidx], nan_map_pooled)[TF])

    ais_n = np.array(ais_n).T
    ais_e = np.array(ais_e).T


    kalman_n = np.array(kalman_n).T
    kalman_e = np.array(kalman_e).T
    kalman_n = kalman_n.reshape(len(kalman_n[0]), len(kalman_n[0][0])) 
    kalman_e = kalman_e.reshape(len(kalman_e[0]), len(kalman_e[0][0])) 

    jcope_n = np.array(jcope_n).T
    jcope_e = np.array(jcope_e).T

    print(f'ais_n shape{ais_n.shape}')
    print(f'ais_e shape{ais_e.shape}')
    print(f'kalman_n shape{kalman_n.shape}')
    print(f'kalman_e shape{kalman_e.shape}')
    print(f'jcope_n shape{jcope_n.shape}')
    print(f'jcope_e shape{jcope_e.shape}')

    x = np.arange(len(ais_n[0]))
    plt.plot(x, jcope_n[0], color='b', label='JCOPE')
    plt.plot(x, ais_n[0], color='r', label='AIS')
    plt.plot(x, kalman_n[0], color='g', label='Kalman')
    for i in range(1, len(ais_n)-1):
        plt.plot(x, jcope_n[i], color='b')
    for i in range(1, len(ais_n)-1):
        plt.plot(x, ais_n[i], color='r')
    for i in range(1, len(ais_n)-1):
        plt.plot(x, kalman_n[i], color='g')
    plt.legend()
    plt.savefig(f'Images/kalmanLog-plotN-{s}.png')
    print(f'saved as Images/kalmanLog-plotN-{s}.png')
    plt.close()

    x = np.arange(len(ais_e[0]))
    plt.plot(x, jcope_e[0], color='b', label='JCOPE')
    plt.plot(x, ais_e[0], color='r', label='AIS')
    plt.plot(x, kalman_e[0], color='g', label='Kalman')
        
    for i in range(1, len(ais_e)-1):
        plt.plot(x, jcope_e[i], color='b')
    for i in range(1, len(ais_e)-1):
        plt.plot(x, ais_e[i], color='r')
    for i in range(1, len(ais_e)-1):
        plt.plot(x, kalman_e[i], color='g')
    plt.legend()
    plt.savefig(f'Images/kalmanLog-plotE-{s}.png')
    print(f'saved as Images/kalmanLog-plotE-{s}.png')



