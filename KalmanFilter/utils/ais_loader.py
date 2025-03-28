import datetime
import numpy as np
import pickle as pkl
import math
import os 
import os.path as osp
import matplotlib.pyplot as plt

from utils.utils import *
from utils.utils_needed_params import *

class AISLoader:
# AISのファイルパスの読み取り
    def __init__(self, year, month, pool_size, use_ais_remove_bad_mmsi=USE_AIS_REMOVE_BAD_MMSI):
        self.year = year
        self.month = month
        self.base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
        self.pathes = None
        #assert pool_size==3, f"Not match pool size (pool_size={pool_size})"
        #path = r"E:\shunsukeE\data\ais\ais_remove_badmmsi_pool3/"  # baged path
        self.removedBroken_default_keys = ['cur1',  'lambda1', 'psi1',\
                    'cur2', 'lambda2', 'psi2', 'N', 'E']
        self.original_default_keys = ['Cur1',  'Lambda1', 'Phi1',\
                    'Cur2', 'Lambda2', 'Phi2', 'N', 'E']
        self.use_ais_remove_bad_mmsi = use_ais_remove_bad_mmsi
        if self.use_ais_remove_bad_mmsi:
            self.path = rf"E:\shunsukeE\data\ais\ais_removedBroken_pool{pool_size}_dummy/" 
            self.keys = self.removedBroken_default_keys
        else:
            #self.path = rf"E:\shunsukeE\data\original_ais_pool{pool_size}_dummy/" 
            self.path = rf"E:\shunsukeE\data\original_ais_pool{pool_size}/" 
            self.keys = self.original_default_keys
        #path = rf"E:\shunsukeE\data\ais\ais_remove_badmmsi_pool1/" 
        print(f"AIS load from {self.path}")

    def set_keys(self, keys):
        self.keys = keys

    def check_fix_keys(self, keys):
        fix_keys = []
        for key in keys:
            if not self.check_key(key):
                fk = self.fix_key(key)
                assert self.check_key(fk), print(f"Unknown Key {key}")
                fix_keys.append(fk)
            else:
                fix_keys.append(key)
        return fix_keys 

    def check_key(self, key):
        if self.use_ais_remove_bad_mmsi:
            return key in self.removedBroken_default_keys
        else:
            return key in self.original_default_keys

    def fix_key(self, key):
        if key[0]=="c":
            return "Cur"+key[-1]
        elif key[0]=="C":
            return "cur"+key[-1]
        elif key[0]=="L":
            return "lambda"+key[-1]
        elif key[0]=="l":
            return "Lambda"+key[-1]
        elif key[0]=="p":
            return "Phi"+key[-1]
        elif key[0]=="P":
            return "psi"+key[-1]
        else:
            return ""

    def load_path(self, keys=None):
        if keys==None:
            keys = self.keys

        keys = self.check_fix_keys(keys)
        pathes = {}
        for key in keys:
            pathes[key] = {}
            
        for day in range(1, 30+1):
            for hour in range(0, 24):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)
                dtidx = date_to_dtidx(self.base_dt, dt)
                
                for key in keys:
                    pathname_base = self.path + f"AisCurr{self.year}{self.month:02}{day:02}{hour:02}"
                    pathname = pathname_base + f"{key}.csv"
                    pathes[key][dtidx] = pathname
        self.pathes = pathes
        return pathes

    def load_ais_day(self, day, keys=None):
        if keys==None:
            keys = self.keys
        keys = self.check_fix_keys(keys)

        def load_ais_data(ais_path):
            ais_data = {}
            for hour in range(24):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)

                dtidx = date_to_dtidx(self.base_dt, dt)
                
                if dtidx in ais_path.keys():
                    try:
                        ais_data[dtidx] = pd.read_csv(ais_path[dtidx], encoding="cp932", header=None)
                        ais_data[dtidx] = ais_data[dtidx].values
                        #ais_data[dtidx] = average_pooling(ais_data[dtidx], pool_size=(pool_size, pool_size))
                    except:
                        print(f'Error load {ais_path[dtidx]}')
            return ais_data

        data = {}
        for key in keys:
            data[key] = load_ais_data(self.pathes[key]) 
        return data

    # TODO Ais4v2ToCur.py 内のAISLoaderと関数名を合わせるため
    def load_cur(self, dtidx, keys=None, use_pool=False):
        return self.load_ais_dtidx(dtidx, keys, use_pool)

    def load_ais_dtidx(self, dtidx, keys=None, use_pool=False):
        if keys==None:
            keys = self.keys
        keys = self.check_fix_keys(keys)

        def load_ais_data(ais_path):
            if dtidx in ais_path.keys():
                #print(f"load from {ais_path[dtidx]}")
                try:
                    ais_data = pd.read_csv(ais_path[dtidx], encoding="cp932", header=None)
                    ais_data = np.array(ais_data.values)
                    # TODO 信頼度の加重平均にすべき
                    if use_pool:
                        ais_data = average_pooling(ais_data, pool_size=(pool_size, pool_size))
                except:
                    assert False, f"Error in loading {ais_path[dtidx]}"
                return ais_data
            else:
                assert False, f"Not found {dtidx} in {ais_path}"

        data = {}
        for key in keys:
            data[key] = load_ais_data(self.pathes[key]) 
        return data
        

def test():
    al = AISLoader(2015, 9)
    keys = ['cur1', 'cur2', 'lambda1', 'lambda2']
    al.set_keys(keys)
    al.load_path(keys)

    # data test
    res = al.load_ais_dtidx(10)
    data = res['cur1']
    print(data)
    print(data.shape)
    plt.imshow(data)
    plt.show()

    data[0][0] = 100
    plt.imshow(data)
    plt.show()

    # load test
    print(f'Testing loading data')
    for i in range(667):
        a = al.load_ais_dtidx(i)
    print(f'Success')
