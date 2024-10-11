import datetime
import numpy as np
import pickle as pkl
import math
import os 
import os.path as osp

from utils.utils import *
from utils.utils_needed_params import *

class AISLoader:
# AISのファイルパスの読み取り
    def __init__(self, year, month):
        self.year = year
        self.month = month
        self.base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
        self.pathes = None
        self.default_keys = ['cur1',  'lambda1', 'phi1',\
                      'cur2', 'lambda2', 'psi2', 'N', 'E']
        self.keys = self.default_keys

    def set_keys(self, keys):
        self.keys = keys

    def load_path(self, keys=None):
        if keys==None:
            keys = self.keys

        path = r"E:\shunsukeE\data\ais\ais_remove_badmmsi/" 
        pathes = {}
        for key in keys:
            pathes[key] = {}
            
        for day in range(1, 30+1):
            for hour in range(0, 24):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)
                dtidx = date_to_dtidx(self.base_dt, dt)

                pathname_base = path + f"AisCurr{self.year}{self.month:02}{day:02}{hour:02}"
                for key in keys:
                    pathname = pathname_base + f"{key}.csv"
                    pathes[key][dtidx] = pathname
        self.pathes = pathes
        return pathes

    def load_path_usepkl(self, keys=None):
        if keys==None:
            keys = self.keys

        pathes = {}
        if 'cur1' in keys:
            ais_cur1_path = pkl.load(open(r'./data/ais_cur1_path.pkl', 'rb'))
            pathes['cur1'] = ais_cur1_path
        if 'cur2' in keys:
            ais_cur2_path = pkl.load(open(r'./data/ais_cur2_path.pkl', 'rb'))
            pathes['cur2'] = ais_cur2_path
        if 'lambda1' in keys:
            ais_lambda1_path = pkl.load(open(r'./data/ais_lambda1_path.pkl', 'rb'))
            pathes['lambda1'] = ais_lambda1_path
        if 'lambda2' in keys:
            ais_lambda2_path = pkl.load(open(r'./data/ais_lambda2_path.pkl', 'rb'))
            pathes['lambda2'] = ais_lambda2_path
        if 'phi1' in keys:
            ais_phi1_path = pkl.load(open(r'./data/ais_phi1_path.pkl', 'rb'))
            pathes['phi1'] = ais_phi1_path
        if 'phi2' in keys:
            ais_phi2_path = pkl.load(open(r'./data/ais_phi2_path.pkl', 'rb'))
            pathes['phi2'] = ais_phi2_path
        if 'n' in keys:
            ais_n_path = pkl.load(open(r'./data/ais_n_path.pkl', 'rb'))
            pathes['n'] = ais_n_path
        if 'e' in keys:
            ais_e_path = pkl.load(open(r'./data/ais_e_path.pkl', 'rb'))
            pathes['e'] = ais_e_path
        if 'd' in keys:
            ais_d_path = pkl.load(open(r'./data/ais_d_path.pkl', 'rb'))
            pathes['d'] = ais_d_path

        self.pathes = pathes
        return pathes

    def load_ais_day(self, day, keys=None):
        if keys==None:
            keys = self.keys

        def load_ais_data(ais_path):
            ais_data = {}
            for hour in range(24):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)

                dtidx = date_to_dtidx(self.base_dt, dt)
                
                if dtidx in ais_path.keys():
                    try:
                        ais_data[dtidx] = pd.read_csv(ais_path[dtidx], encoding="cp932", header=None)
                        ais_data[dtidx] = ais_data[dtidx].values
                        ais_data[dtidx] = average_pooling(ais_data[dtidx], pool_size=(pool_size, pool_size))
                    except:
                        print(f'Error load {ais_path[dtidx]}')
            return ais_data

        data = {}
        for key in keys:
            data[key] = load_ais_data(self.pathes[key]) 
        return data

    # Ais4v2ToCur.py 内のAISLoaderと関数名を合わせるため
    def load_cur(self, dtidx, keys=None, use_pool=True):
        return self.load_ais_dtidx(dtidx, keys, use_pool)

    def load_ais_dtidx(self, dtidx, keys=None, use_pool=True):
        if keys==None:
            keys = self.keys

        def load_ais_data(ais_path):
            ais_data = []
            if dtidx in ais_path.keys():
                #print(f"load from {ais_path[dtidx]}")
                try:
                    ais_data = pd.read_csv(ais_path[dtidx], encoding="cp932", header=None)
                    ais_data = ais_data.values
                    # TODO 信頼度の加重平均にすべき
                    if use_pool:
                        ais_data = average_pooling(ais_data, pool_size=(pool_size, pool_size))
                except:
                    print(f'Error load {ais_path[dtidx]}')
            return ais_data

        data = {}
        for key in keys:
            data[key] = load_ais_data(self.pathes[key]) 
        return data

if __name__=='__main__':
    al = AISLoader(2015, 9)
    keys = ['lambda1']
    al.load_path(keys=keys)
    #path_cur1 = al.pathes['cur1']
    #print(f'path num: {len(path_cur1)}')
    #print(f'expample keys:{path_cur1.keys()},\n val0: {path_cur1[0]}')

    #data = al.load_ais_day(1)
    print(f'Testing loading data')
    for i in range(667):
        a = al.load_ais_dtidx(i)
    print(f'Success')
