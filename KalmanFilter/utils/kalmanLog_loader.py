
import datetime
import numpy as np
import pickle as pkl
import math
import os 
import os.path as osp

from utils.utils import *

class KalmanLogLoader:
    def __init__(self, year, month, nslice):
        self.year = year
        self.month = month
        self.nslice = nslice
        self.base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
        self.log_path = None
        self.keys = ['X', 'XCur', 'Z', 'H', 'P', 'F', 'K',\
                     'R', 'JCOPE', 'JCOPECur', 'Target', 'TF',\
                     'var', 'error']
        # Target is available ais data, TF is target data in sliced data
        
    def set_path(self, path):
        self.log_path = path

    def load_kalmanLog_day_hour(self, day, hour, s, keys=None):
        if keys==None:
            keys = self.keys

        data = {}
        for key in keys:
            if key=='TF':
                path = osp.join(self.log_path, fr'Targets20150901-{s}.pkl')
                if osp.isfile(path):
                    data['TF'] = pkl.load(open(path, 'rb'))
                else:
                    print(f'is file error: day, hour:{day}, {hour}')
                    data['TF'] = None
                continue        
            if not key in ['error', 'var']:
                path = osp.join(self.log_path, fr'saver{key}201509{day:02}{hour:02}-{s}.pkl')
            else:
                path = osp.join(self.log_path, r"ais_files/" + fr"{key}201509{day:02}{hour:02}-{s}.pkl")
            if osp.isfile(path):
                data[key] = pkl.load(open(path, 'rb'))
            else:
                print(f'is file error: day, hour:{day}, {hour}')
                data[key] = None
                input('is file error')
        return data

    def load_kalmanLog_day(self, day, s, keys=None):
        if keys==None:
            keys = self.keys

        def load_kalmanLog(key):
            kalmanLog = {}
            #start_hour = 0 if day == 1 and key!='K' else 1
            start_hour = 1
            for hour in range(start_hour, 23):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)
                dtidx = date_to_dtidx(self.base_dt, dt)
                if not key in ['error', 'var']:
                    path = osp.join(self.log_path, fr'saver{key}201509{day:02}{hour:02}-{s}.pkl')
                else:
                    path = osp.join(self.log_path, r"ais_files/" + fr"{key}201509{day:02}{hour:02}-{s}.pkl")
                kalmanLog[dtidx] = pkl.load(open(path, 'rb'))
            return kalmanLog 

        data = {}
        for key in keys:
            if key=='TF':
                path = osp.join(self.log_path, fr'Targets201509{day:02}-{s}.pkl')
                data['TF'] = pkl.load(open(path, 'rb'))
                continue        
            data[key] = load_kalmanLog(key) 
        return data

    def load_kalmanLog_day(self, day, s, keys=None):
        if keys==None:
            keys = self.keys

        def load_kalmanLog(key):
            kalmanLog = {}
            #start_hour = 0 if day == 1 and key!='K' else 1
            start_hour = 1
            for hour in range(start_hour, 23):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)
                dtidx = date_to_dtidx(self.base_dt, dt)
                if not key in ['error', 'var']:
                    path = osp.join(self.log_path, fr'saver{key}201509{day:02}{hour:02}-{s}.pkl')
                else:
                    path = osp.join(self.log_path, r"ais_files/" + fr"{key}201509{day:02}{hour:02}-{s}.pkl")
                kalmanLog[dtidx] = pkl.load(open(path, 'rb'))
            return kalmanLog 

        data = {}
        for key in keys:
            if key=='TF':
                path = osp.join(self.log_path, fr'Targets201509{day:02}-{s}.pkl')
                data['TF'] = pkl.load(open(path, 'rb'))
                continue        
            data[key] = load_kalmanLog(key) 
        return data

if __name__=='__main__':
    kl = KalmanLogLoader(2015, 9, 20)
    log_path = r"E:/shunsukeE/result/kalman-enough_data/" 
    kl.set_path(log_path)

    print(f'Testing loading data')

    keys = ['TF', 'X', 'Z']
    for day in range(1, 29):
        a = kl.load_kalmanLog_day(day, 0, keys=keys)

    print(f'Successed')
