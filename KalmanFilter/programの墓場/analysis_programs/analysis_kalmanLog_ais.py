import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from utils import * 

year = 2015
month = 9
n_day = 7 #nday_month(month)
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)

print(f'Loading data')
al = AISLoader(2015, 9)
ais_keys = ['n', 'e', 'd']
al.load_path(keys=ais_keys)

kl = KalmanLogLoader(2015, 9, 20)
log_path = r"E:\shunsukeE\result\kalman-all-Q001"
kalman_keys = ['X']
kl.set_path(log_path)

jl = JCOPELoader(2015, 9)
path_jcope = fr'E:\shunsukeE\data\eas2'
jl.load_path(path_jcope)

for s in range(0, 1):
    ais_n = []
    ais_e = []
    ais_n2 = []
    ais_e2 = []
    for day in range(1, n_day-1):
        print(f's"{s}, day:{day}')
        # TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']
        TF = np.array([True]*9808)
        
        start_hour = 0 if day == 1 else 1
        data = al.load_ais_day(day)
        for hour in range(start_hour, 23):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue

            ais_n.append(kurosio_filter(data['n'][dtidx][0], nan_map)[TF])
            ais_e.append(kurosio_filter(data['e'][dtidx][0], nan_map)[TF])

        for hour in range(start_hour, 23):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue
            path = f"{log_path}/ais_files/AisCurr{year}{month:02}{day:02}{hour:02}X.csv"
            dataN = pd.read_csv(path, encoding="cp932", header=None)
            dataN = dataN.values
            ais_n2.append(kurosio_filter(dataN, nan_map)[TF])
            path = f"{log_path}/ais_files/AisCurr{year}{month:02}{day:02}{hour:02}Y.csv"
            dataE = pd.read_csv(path, encoding="cp932", header=None)
            dataE = dataE.values
            ais_e2.append(kurosio_filter(dataE, nan_map)[TF])

    ais_n = np.array(ais_n).T
    ais_e = np.array(ais_e).T
    ais_n2 = np.array(ais_n2).T
    ais_e2 = np.array(ais_e2).T
    #tf = ais_d<10**4
    #ais_n[tf] = np.nan
    #ais_e[tf] = np.nan



    print(f'ais_n shape{ais_n.shape}')
    print(f'ais_e shape{ais_e.shape}')

    x = np.arange(len(ais_n[0]))
    plt.plot(x, ais_n[0], color='r', label='not use shipvar')
    plt.plot(x, ais_n2[0], color='g', label='use shipvar')

    for i in range(1, len(ais_n)-1):
        plt.plot(x, ais_n[i], color='r')
    for i in range(1, len(ais_n2)-1):
        plt.plot(x, ais_n2[i], color='g')

    plt.legend()
    plt.savefig(f'Images/AIStest2N-{s}.png')
    print(f'saved as Images/AIStest2N-{s}.png')
    plt.close()

    x = np.arange(len(ais_e[0]))
    plt.plot(x, ais_e[0], color='r', label='not use shipvar')
    plt.plot(x, ais_e2[0], color='g', label='use shipvar')

    for i in range(1, len(ais_n)-1):
        plt.plot(x, ais_e[i], color='r')
    for i in range(1, len(ais_e2)-1):
        plt.plot(x, ais_e2[i], color='g')

    plt.legend()
    plt.savefig(f'Images/AIStest2E-{s}.png')
    print(f'saved as Images/AIStest2E-{s}.png')
    plt.close()


