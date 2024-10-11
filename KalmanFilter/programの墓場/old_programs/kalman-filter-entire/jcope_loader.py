import datetime
import numpy as np
import pickle as pkl
import math
import os 
import os.path as osp

from entire_utils import *
from entire_kf_params import *

class JCOPELoader:
    def __init__(self, year, month):
        self.year = year
        self.month = month
        self.base_dt = datetime.datetime(year, month, 1, 0, 0, 0)
        
    def load_path(self, path_jcope):
# JCOPEのファイルパスの読み取り
        jcope_n_path = {}
        jcope_e_path = {}
        n_day = nday_month(self.month)
        for day in range(1, n_day+1):
            for hour in range(24):
                dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)
                dtidx = date_to_dtidx(self.base_dt, dt)
                jcope_n_path[dtidx] = osp.join(path_jcope, f'{self.year}{self.month:02}{day:02}{hour:02}', "N.csv")
                jcope_e_path[dtidx] = osp.join(path_jcope, f'{self.year}{self.month:02}{day:02}{hour:02}', "E.csv")
        pathes = { 'jcope_n': jcope_n_path,
                    'jcope_e': jcope_e_path}
        self.pathes = pathes
        return pathes

    def load_jcope_day(self, day):
        jcope_n = {}
        jcope_e = {}

        for hour in range(24):
            dt = datetime.datetime(self.year, self.month, day, hour, 0, 0)
            dtidx = date_to_dtidx(self.base_dt, dt)
            jcope_n[dtidx] = []
            jcope_e[dtidx] = []
            path = self.pathes['jcope_n'][dtidx]
            try:
                n = pd.read_csv(path, encoding="cp932", header=None)
                n = n.values
                jcope_n[dtidx] = n
            except:
                print(f'Error load {path}')

            path = self.pathes['jcope_e'][dtidx]
            try:
                e = pd.read_csv(path, encoding="cp932", header=None)
                e = e.values
                jcope_e[dtidx] = e
            except:
                print(f'Error load {path}')

        return jcope_n, jcope_e

if __name__=='__main__':
    jl = JCOPELoader(2015, 9)
    path_jcope = fr'E:\shunsukeE\data\eas2'
    jl.load_path(path_jcope)
    day = 1
    jcope_n, jcope_e = jl.load_jcope_day(day)
    print(jcope_n.keys())
    print(jcope_e.keys())
    
