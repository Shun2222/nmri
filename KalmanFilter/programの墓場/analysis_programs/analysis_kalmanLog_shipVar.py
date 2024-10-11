import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ais_loader import AISLoader
from jcope_loader import JCOPELoader 
from kalmanLog_loader import KalmanLogLoader
from utils import * 

year = 2015
month = 9
n_day = 2 #nday_month(month)
base_dt = datetime.datetime(year, month, 1, 0, 0, 0)

print(f'Loading data')

kl = KalmanLogLoader(2015, 9, 400)
log_path = r"E:\shunsukeE\result\kalman-shipVar-time-test"
kalman_keys = ['var', 'error']
kl.set_path(log_path)


for s in range(0, 1):
    ship_vars = {}
    ship_errors = {}
    ship_error_num = {}
    
    day = n_day-1
    TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']

    data = kl.load_kalmanLog_day_hour(day, 23, s, keys=kalman_keys)
    all_mmsi = data['var'].keys()
    all_mmsi2 = data['error'].keys()
    # all_mmsi = [431000848]
    # all_mmsi2 = [431000848]
    print(f'Num mmsi: {len(all_mmsi)}') 
    
    for mmsi in all_mmsi:
        ship_vars[mmsi] = []
        ship_errors[mmsi] = []
        ship_error_num[mmsi] = []
        
    for day in range(1, n_day):
        print(f's"{s}, day:{day}')
        TF = kl.load_kalmanLog_day(1, s, keys=['TF'])['TF']

        for hour in range(0, 24):
            dt = datetime.datetime(year, month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            data = kl.load_kalmanLog_day_hour(day, hour, s, keys=kalman_keys)
            data['var'].keys()
            
            for mmsi in all_mmsi:
                if not mmsi in data['var'].keys():
                    ship_vars[mmsi].append(np.nan)
                else:
                    ship_vars[mmsi].append(data['var'][mmsi])

            for mmsi in all_mmsi2:        
                if not mmsi in data['error'].keys():
                    ship_errors[mmsi].append(np.nan)
                    ship_error_num[mmsi].append(np.nan)
                else:
                    ship_errors[mmsi].append(np.max(data['error'][mmsi]))
                    ship_error_num[mmsi].append(len(data['error'][mmsi]))
                    v = data['error'][mmsi]
                    print(f'error: {v}')

    for mmsi in all_mmsi:
        ship_var = np.array(ship_vars[mmsi])
        print(f'ship_var shape{ship_var.shape}')

        x = np.arange(len(ship_var))
        plt.plot(x, ship_var, label=f'{mmsi}')
    plt.legend()
    plt.savefig(f'Images/var-{s}.png')
    print(f'saved as Images/var-{s}.png')
    plt.close()
    
    for mmsi in all_mmsi2:
        ship_error = np.array(ship_errors[mmsi])
        x = np.arange(len(ship_error))
        plt.plot(x, ship_error, label=f'{mmsi}')
    plt.legend()
    plt.savefig(f'Images/error-{s}.png')
    print(f'saved as Images/error-{s}.png')
    plt.close()

    for mmsi in all_mmsi2:
        num = np.array(ship_error_num[mmsi])
        x = np.arange(len(num))
        plt.plot(x, num, label=f'{mmsi}')
    plt.legend()
    plt.savefig(f'Images/error_num-{s}.png')
    print(f'saved as Images/error_num-{s}.png')
    plt.close()

