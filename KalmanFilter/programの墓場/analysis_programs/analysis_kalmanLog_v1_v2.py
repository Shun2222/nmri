import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from utils import * 

year = 2015
month = 9
n_day = 3#nday_month(month)
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
_NHalf = 400
_M = 1
_N = 2 * _NHalf + 1

ais_keys = ['cur1', 'cur2', 'lambda1', 'lambda2']
al = AISLoader(year, month)
al.load_path(keys=ais_keys)

kl = KalmanLogLoader(2015, 9, 20)
#log_path = r"E:/shunsukeE/result/kalman-enough_data-AIStest2/" 
log_path = r"E:\shunsukeE\result\kalman-shipVar-time-test2"
kalman_keys = ['XCur', 'JCOPECur', 'Z']
kl.set_path(log_path)
save_dir = 'Images/test/'

def get_z(ais_cur1, ais_cur2, dtidx, isTarget):
    if not dtidx in ais_cur1.keys() or not dtidx in ais_cur2.keys():
        return np.array([]) 
    min_value = 0.0 #1/1e10
    ais_cur1_dt = kurosio_filter(ais_cur1[dtidx][0], nan_map)[isTarget] # 時刻0のais data
    ais_cur2_dt = kurosio_filter(ais_cur2[dtidx][0], nan_map)[isTarget] # 時刻0のais data
    ais_cur12_dt = np.concatenate([ais_cur1_dt, ais_cur2_dt])
    ais_cur12_dt = np.concatenate([ais_cur12_dt, [1]])
    ais_cur12_dt = ais_cur12_dt.reshape(len(ais_cur12_dt), 1)
    return ais_cur12_dt

for s in range(0, 1):
    ais_cur = []
    ais_lambda = []
    kalman_cur = []
    jcope_cur = []
    for day in range(1, n_day-1):
        TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']

        #start_hour = 0 if day == 1 else 1
        start_hour = 1


        data = kl.load_kalmanLog_day(day, s, keys=kalman_keys)
        ais_data = al.load_ais_day(day)
        ais_cur1, ais_cur2, ais_lambda1, ais_lambda2 = [ais_data[key] for key in ais_keys]

        for hour in range(start_hour, 23):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)

            z = get_z(ais_cur1, ais_cur2, dtidx, TF)
            ais_cur.append(z)

            l = get_z(ais_lambda1, ais_lambda2, dtidx, TF)
            ais_lambda.append(l)

            nonZero = z!=0.0
            nonZero = nonZero[:_NHalf] & nonZero[_NHalf:-1]
            nonZero = np.concatenate([nonZero, nonZero])
            nonZero = np.concatenate([nonZero, [[True]]])
            notNan = (~np.isnan(z)) & (nonZero)

            a = np.zeros((len(notNan), 1)) * np.nan
            print(len(data['XCur'][dtidx]))
            if np.sum(notNan)!=0:
                a[notNan] = data['XCur'][dtidx]
            kalman_cur.append(a)

            a = np.zeros((len(notNan), 1)) * np.nan
            a[notNan] = data['JCOPECur'][dtidx]
            jcope_cur.append(a)

    ais_cur = np.array(ais_cur).T
    ais_lambda = np.array(ais_lambda).T
    kalman_cur = np.array(kalman_cur).T
    jcope_cur = np.array(jcope_cur).T

    print(ais_cur.shape)
    x = np.arange(len(ais_cur[0][0]))
    plt.plot(x, jcope_cur[0][0], color='b', label='JCOPE')
    plt.plot(x, ais_cur[0][0], color='r', label='AIS')
    plt.plot(x, kalman_cur[0][0], color='g', label='Kalman')
    for i in range(1, _NHalf):
        plt.plot(x, ais_cur[0][i], color='r')
        plt.plot(x, jcope_cur[0][i], color='b')
        plt.plot(x, kalman_cur[0][i], color='g')
    plt.legend()
    plt.savefig(f'{save_dir}AIStest2-v1-{s}.png')
    plt.close()

    plt.plot(x, jcope_cur[0][0], color='b', label='JCOPE')
    plt.plot(x, ais_cur[0][0], color='r', label='AIS')
    plt.plot(x, kalman_cur[0][0], color='g', label='Kalman')
    for i in range(_NHalf, len(ais_cur[0])):
        plt.plot(x, ais_cur[0][i], color='r')
        plt.plot(x, jcope_cur[0][i], color='b')
        plt.plot(x, kalman_cur[0][i], color='g')
    plt.legend()
    plt.savefig(f'{save_dir}AIStest2-v2-{s}.png')
    plt.close()

    plt.plot(x, ais_lambda[0][0], color='r', label='ais lambda')
    for i in range(1, _NHalf):
        plt.plot(x, ais_lambda[0][i], color='r')
    plt.legend()
    plt.savefig(f'{save_dir}AIS-LAMBDA-test2-v1-{s}.png')
    plt.close()

    plt.plot(x, ais_lambda[0][0], color='r', label='ais lambda')
    for i in range(_NHalf, len(ais_lambda[0])):
        plt.plot(x, ais_lambda[0][i], color='r')
    plt.legend()
    plt.savefig(f'{save_dir}AIS-LAMBDA-test2-v2-{s}.png')
    plt.close()
    print(f'Saved picture in {save_dir}')
