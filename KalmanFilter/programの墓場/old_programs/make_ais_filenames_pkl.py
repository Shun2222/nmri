import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os 
import os.path as osp
import re
import pickle as pkl
from tqdm import tqdm
from utils import *
from kf_params import *
import logger
import printManager as pm

pm.printline('Setting parameter')
save_path = r'./data'
path_ais = fr'E:\shunsukeE\data\ais'
year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) 
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)

ais_cur1_path = {} #偏流1
ais_cur2_path = {} #偏流2
ais_lambda1_path = {} #固有値1
ais_lambda2_path = {} #固有値2
ais_phi1_path = {} #固有ベクトルの方向1
ais_phi2_path = {} #固有ベクトルの方向2
ais_n_path = {} 
ais_e_path = {} 
ais_d_path = {} 

dt = datetime.datetime(dt_year, dt_month, n_day, 23, 0, 0)
max_dtidx = date_to_dtidx(base_dt, dt) 
ais_cur1_ndata = np.zeros(max_dtidx) #偏流1
ais_cur2_ndata = np.zeros(max_dtidx) #偏流2
ais_lambda1_ndata = np.zeros(max_dtidx) #固有値1
ais_lambda2_ndata = np.zeros(max_dtidx) #固有値2
ais_phi1_ndata = np.zeros(max_dtidx) #固有ベクトルの方向1
ais_phi2_ndata = np.zeros(max_dtidx) #固有ベクトルの方向2
ais_n_ndata = np.zeros(max_dtidx)
ais_e_ndata = np.zeros(max_dtidx) 
ais_d_ndata = np.zeros(max_dtidx)

patterns = [f'(\w+){month:02}..[0-9][0-9]Cur1.csv', 
            f'(\w+){month:02}..[0-9][0-9]Cur2.csv', 
            f'(\w+){month:02}..[0-9][0-9]Lambda1.csv',
            f'(\w+){month:02}..[0-9][0-9]Lambda2.csv',
            f'(\w+){month:02}..[0-9][0-9]Phi1.csv',
            f'(\w+){month:02}..[0-9][0-9]Phi2.csv',
            f'(\w+){month:02}..[0-9][0-9]N.csv',
            f'(\w+){month:02}..[0-9][0-9]E.csv',
            f'(\w+){month:02}..[0-9][0-9]D.csv']

pm.printline('Load files')
files = os.listdir(path_ais)
dirs = [f for f in files if os.path.isdir(path_ais)]
for d in dirs:
    path2 = osp.join(path_ais, d, 'log')
    files = os.listdir(path2)
    filenames = [f for f in files if os.path.isfile(osp.join(path2, f))]
    pm.printline(f'Checking in {path2}')
    for f in filenames:
        if re.match(patterns[0], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_cur1_ndata[dtidx] += 1 
            if not dtidx in ais_cur1_path.keys():
                ais_cur1_path[dtidx] = []
                ais_cur1_path[dtidx].append(f_path)
            else:
                ais_cur1_path[dtidx].append(f_path)

        if re.match(patterns[1], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_cur2_ndata[dtidx] += 1 
            if not dtidx in ais_cur2_path.keys():
                ais_cur2_path[dtidx] = []
                ais_cur2_path[dtidx].append(f_path)
            else:
                ais_cur2_path[dtidx].append(f_path)


        if re.match(patterns[2], f):
            f_path = osp.join(path2, f)
            day = int(f[-15:-13])
            hour = int(f[-13:-11])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_lambda1_ndata[dtidx] += 1 
            if not dtidx in ais_lambda1_path.keys():
                ais_lambda1_path[dtidx] = []
                ais_lambda1_path[dtidx].append(f_path)
            else:
                ais_lambda1_path[dtidx].append(f_path)    

        if re.match(patterns[3], f):
            f_path = osp.join(path2, f)
            day = int(f[-15:-13])
            hour = int(f[-13:-11])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_lambda2_ndata[dtidx] += 1 
            if not dtidx in ais_lambda2_path.keys():
                ais_lambda2_path[dtidx] = []
                ais_lambda2_path[dtidx].append(f_path)
            else:
                ais_lambda2_path[dtidx].append(f_path)    

        if re.match(patterns[4], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_phi1_ndata[dtidx] += 1 
            if not dtidx in ais_phi1_path.keys():
                ais_phi1_path[dtidx] = []
                ais_phi1_path[dtidx].append(f_path)
            else:
                ais_phi1_path[dtidx].append(f_path)    

        if re.match(patterns[5], f):
            f_path = osp.join(path2, f)
            day = int(f[-12:-10])
            hour = int(f[-10:-8])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_phi2_ndata[dtidx] += 1 
            if not dtidx in ais_phi2_path.keys():
                ais_phi2_path[dtidx] = []
                ais_phi2_path[dtidx].append(f_path)
            else:
                ais_phi2_path[dtidx].append(f_path)    

        if re.match(patterns[6], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_n_ndata[dtidx] += 1 
            if not dtidx in ais_n_path.keys():
                ais_n_path[dtidx] = []
                ais_n_path[dtidx].append(f_path)
            else:
                ais_n_path[dtidx].append(f_path)


        if re.match(patterns[7], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_e_ndata[dtidx] += 1 
            if not dtidx in ais_e_path.keys():
                ais_e_path[dtidx] = []
                ais_e_path[dtidx].append(f_path)
            else:
                ais_e_path[dtidx].append(f_path)


        if re.match(patterns[8], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            ais_d_ndata[dtidx] += 1 
            if not dtidx in ais_d_path.keys():
                ais_d_path[dtidx] = []
                ais_d_path[dtidx].append(f_path)
            else:
                ais_d_path[dtidx].append(f_path)    

pm.printline('Saving result')
pkl.dump(ais_cur1_path, open(osp.join(save_path, 'ais_cur1_path.pkl'), 'wb'))
pkl.dump(ais_cur2_path, open(osp.join(save_path,'ais_cur2_path.pkl'), 'wb'))
pkl.dump(ais_lambda1_path, open(osp.join(save_path,'ais_lambda1_path.pkl'), 'wb'))
pkl.dump(ais_lambda2_path, open(osp.join(save_path,'ais_lambda2_path.pkl'), 'wb'))
pkl.dump(ais_phi1_path, open(osp.join(save_path,'ais_phi1_path.pkl'), 'wb'))
pkl.dump(ais_phi2_path, open(osp.join(save_path,'ais_phi2_path.pkl'), 'wb'))
pkl.dump(ais_n_path, open(osp.join(save_path,'ais_n_path.pkl'), 'wb'))
pkl.dump(ais_e_path, open(osp.join(save_path,'ais_e_path.pkl'), 'wb'))
pkl.dump(ais_d_path, open(osp.join(save_path,'ais_d_path.pkl'), 'wb'))

pkl.dump(ais_cur1_ndata, open(osp.join(save_path, 'ais_cur1_ndata.pkl'), 'wb'))
pkl.dump(ais_cur2_ndata, open(osp.join(save_path,'ais_cur2_ndata.pkl'), 'wb'))
pkl.dump(ais_lambda1_ndata, open(osp.join(save_path,'ais_lambda1_ndata.pkl'), 'wb'))
pkl.dump(ais_lambda2_ndata, open(osp.join(save_path,'ais_lambda2_ndata.pkl'), 'wb'))
pkl.dump(ais_phi1_ndata, open(osp.join(save_path,'ais_phi1_ndata.pkl'), 'wb'))
pkl.dump(ais_phi2_ndata, open(osp.join(save_path,'ais_phi2_ndata.pkl'), 'wb'))
pkl.dump(ais_n_ndata, open(osp.join(save_path,'ais_n_ndata.pkl'), 'wb'))
pkl.dump(ais_e_ndata, open(osp.join(save_path,'ais_e_ndata.pkl'), 'wb'))
pkl.dump(ais_d_ndata, open(osp.join(save_path,'ais_d_ndata.pkl'), 'wb'))

plt.bar(np.arange(24), ais_cur1_ndata[:24], color='b', label='cur1')
plt.bar(np.arange(24), ais_cur2_ndata[:24], color='r', label='cur2')
plt.xlabel('dtidx')
plt.ylabel('Num path')
plt.legend()
plt.savefig(osp.join(save_path, 'cur_num_path_firstday.png'))
plt.close()

plt.bar(np.arange(max_dtidx), ais_cur1_ndata, color='b', label='cur1')
plt.bar(np.arange(max_dtidx), ais_cur2_ndata, color='r', label='cur2')
plt.xlabel('dtidx')
plt.ylabel('Num path')
plt.legend()
plt.savefig(osp.join(save_path, 'cur_num_path.png'))
plt.close()

plt.bar(np.arange(max_dtidx), ais_lambda1_ndata, color='b', label='lambda1')
plt.bar(np.arange(max_dtidx), ais_lambda2_ndata, color='r', label='lambda2')
plt.xlabel('dtidx')
plt.ylabel('Num path')
plt.legend()
plt.savefig(osp.join(save_path, 'lambda_num_path.png'))
plt.close()

plt.bar(np.arange(max_dtidx), ais_phi1_ndata, color='b', label='phi1')
plt.bar(np.arange(max_dtidx), ais_phi2_ndata, color='r', label='phi2')
plt.xlabel('dtidx')
plt.ylabel('Num path')
plt.legend()
plt.savefig(osp.join(save_path, 'phi_num_path.png'))
plt.close()

no_data = np.zeros(max_dtidx) 
no_data[ais_cur1_ndata==0] = 1 
plt.bar(np.arange(24), no_data[:24], color='b', label='cur1')
no_data = np.zeros(max_dtidx) 
no_data[ais_cur2_ndata==0] = 1 
plt.bar(np.arange(24), no_data[:24], color='r', label='cur2')
plt.xlabel('dtidx')
plt.ylabel('Is not exist path')
plt.legend()
plt.savefig(osp.join(save_path, 'cur_not_exist_path_firstday.png'))
plt.close()

no_data = np.zeros(max_dtidx) 
no_data[ais_cur1_ndata==0] = 1 
plt.bar(np.arange(max_dtidx), no_data, color='b', label='cur1')
no_data = np.zeros(max_dtidx) 
no_data[ais_cur2_ndata==0] = 1 
plt.bar(np.arange(max_dtidx), no_data, color='r', label='cur2')
plt.xlabel('dtidx')
plt.ylabel('Is not exist path')
plt.legend()
plt.savefig(osp.join(save_path, 'cur_not_exist_path.png'))
plt.close()

no_data = np.zeros(max_dtidx) 
no_data[ais_lambda1_ndata==0] = 1 
plt.bar(np.arange(max_dtidx), no_data, color='b', label='lambda1')
no_data = np.zeros(max_dtidx) 
no_data[ais_lambda2_ndata==0] = 1 
plt.bar(np.arange(max_dtidx), no_data, color='r', label='lambda2')
plt.xlabel('dtidx')
plt.ylabel('Is not exist path')
plt.legend()
plt.savefig(osp.join(save_path, 'lambda_not_exist_path.png'))
plt.close()

no_data = np.zeros(max_dtidx) 
no_data[ais_phi1_ndata==0] = 1 
plt.bar(np.arange(max_dtidx), no_data, color='b', label='phi1')
no_data = np.zeros(max_dtidx) 
no_data[ais_phi2_ndata==0] = 1 
plt.bar(np.arange(max_dtidx), no_data, color='r', label='phi2')
plt.xlabel('dtidx')
plt.ylabel('Is not exist path')
plt.legend()
plt.savefig(osp.join(save_path, 'phi_not_exist_path.png'))
plt.close()
