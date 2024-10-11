import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from utils import * 

year = 2015
month = 9
n_day = 3 #nday_month(month)
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)

print(f'Loading data')
ais_keys = ['Lambda1', 'Lambda2']

kl = KalmanLogLoader(2015, 9, 20)
log_path = r"E:\shunsukeE\result\kalman-shipVar-time-test4"
kalman_keys = ['X', 'Z', 'Target']
kl.set_path(log_path)



for s in range(0, 1):
    ais_n = []
    ais_e = []
    ais_d = []
    kalman_n = []
    kalman_e = []
    jcope_n = []
    jcope_e = []
    for day in range(1, n_day-1):
        print(f's"{s}, day:{day}')
        TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']

        start_hour = 0 if day == 1 else 1
        for hour in range(start_hour, 23):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            if dtidx==0:
                continue
            path = f"{log_path}/ais_files/AisCurr{year}{month:02}{day:02}{hour:02}{ais_keys[0]}.csv"
            dataN = pd.read_csv(path, encoding="cp932", header=None)
            dataN = dataN.values
            ais_n.append(kurosio_filter(dataN, nan_map)[TF])
            path = f"{log_path}/ais_files/AisCurr{year}{month:02}{day:02}{hour:02}{ais_keys[1]}.csv"
            dataE = pd.read_csv(path, encoding="cp932", header=None)
            dataE = dataE.values
            ais_e.append(kurosio_filter(dataE, nan_map)[TF])


    ais_n = np.array(ais_n).T
    ais_e = np.array(ais_e).T


    print(f'ais_n shape{ais_n.shape}')
    print(f'ais_e shape{ais_e.shape}')

    x = np.arange(len(ais_n[0]))
    plt.plot(x, ais_n[0], color='r', label=f'{ais_keys[0]}')
    plt.plot(x, ais_n[0], color='g', label=f'{ais_keys[1]}')
    for i in range(1, len(ais_n)-1):
        plt.plot(x, ais_n[i], color='r')   
    for i in range(1, len(ais_e)-1):
        plt.plot(x, ais_e[i], color='g')
    plt.legend()
    plt.savefig(f'Images/kalmanLog-plot-{s}.png')
    print(f'saved as Images/kalmanLog-plot-{s}.png')
    plt.close()


