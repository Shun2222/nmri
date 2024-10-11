import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
import tqdm
import pickle as pkl
import math
import os 
import os.path as osp
import re

from utils import *
from kf_params import *



year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month) - 1
base_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
done_ship = []

path_jcope = fr'E:\shunsukeE\data\eas2'
path_ais = fr'E:\shunsukeE\data\ais'
path_ship = path = r"E:\shunsukeE\data\shiplog/"

def diff_ais_jcope_ship(ais, jcope, shipLog, use_d=True):
    ais_n = ais[0]
    ais_e = ais[1]
    jcope_n = jcope[0]
    jcope_e = jcope[1]
    #for i in range(len(shipLog)):
    diff_ais_jcope = []
    diff_ais_ship = []
    diff_jcope_ship = []
    dtidxs = []
    grids = []

    for i in range(len(shipLog)):
        time = shipLog['DtIdx'].values[i]
        grid0 = shipLog['Grid0'].values[i]
        grid1 = shipLog['Grid1'].values[i]
        curN = shipLog['CurN'].values[i]
        curE = shipLog['CurE'].values[i]
        dtidxs.append(time)
        dtidx = time
        grids.append([grid0, grid1])

        jcope_curN = np.nan
        jcope_curE = np.nan
        if time >= 0:
            if time in jcope_n.keys(): 
                if len(jcope_n[time])!=0: 
                    if grid0<len(jcope_n[dtidx]) and grid1<len(jcope_n[dtidx][0]):
                        jcope_curN = jcope_n[time][grid0][grid1]
            
            if time in jcope_e.keys(): 
                if len(jcope_e[time])!=0:   
                    if grid0<len(jcope_e[dtidx]) and grid1<len(jcope_e[dtidx][0]):
                        jcope_curE = jcope_e[time][grid0][grid1]

        ais_curN = np.nan
        ais_curE = np.nan
        if time in ais_n.keys(): 
            if len(ais_n[time])!=0: 
                if grid0<len(ais_n[dtidx][0]) and grid1<len(ais_n[dtidx][0][0]):
                    if ais_d[time][0][grid0][grid1]>=1e5 or not use_d:
                        ais_curN = ais_n[time][0][grid0][grid1]
        if time in ais_e.keys():
            if len(ais_e[time])!=0: 
                if grid0<len(ais_e[dtidx][0]) and grid1<len(ais_e[dtidx][0][0]):
                    if ais_d[dtidx][0][grid0][grid1]>=1e5 or not use_d:
                        ais_curE = ais_e[time][0][grid0][grid1]
        
        diff_ais_jcope.append([ais_curN-jcope_curN, ais_curE-jcope_curE])
        diff_ais_ship.append([ais_curN-curN, ais_curE-curE])
        diff_jcope_ship.append([jcope_curN-curN, jcope_curE-curE])    


        #print('-------------------------')
        #print(f'Time: {time}, grid: ({grid0}, {grid1})')
        #print(f'Diff ais and jcope: {ais_curN-jcope_curN:.2}, {ais_curE-jcope_curE:.2}')
        #print(f'Diff ais and ship: {ais_curN-curN:.2}, {ais_curE-curE:.2}')
        #print(f'Diff jcope and ship: {jcope_curN-curN:.2}, {jcope_curE-curE:.2}')
        #print('-------------------------')

    diff_results = pd.DataFrame([])
    diff_results['DtIdx'] = dtidxs

    grids_np = np.array(grids)
    diff_results['Grid0'] = grids_np.T[0]
    diff_results['Grid1'] = grids_np.T[1]

    a = np.array(diff_ais_jcope)
    diff_results['AIS-JCOPE_N'] = a.T[0] 
    diff_results['AIS-JCOPE_E'] = a.T[1] 

    a = np.array(diff_ais_ship)
    diff_results['AIS-Ship_N'] = a.T[0] 
    diff_results['AIS-Ship_E'] = a.T[1] 

    a = np.array(diff_jcope_ship)
    diff_results['JCOPE-Ship_N'] = a.T[0] 
    diff_results['JCOPE-Ship_E'] = a.T[1] 

    return diff_results

def print_diff_ais_jcope_ship(diff_results):

    tf = diff_results['AIS-JCOPE_N']==diff_results['AIS-JCOPE_N']
    data = diff_results[tf]
    if len(data)==0:
        diff_ais_jcope_n = np.nan
    else:
        diff_ais_jcope_n = np.sqrt(np.sum(data['AIS-JCOPE_N']*data['AIS-JCOPE_N'])/len(data))
    tf = diff_results['AIS-JCOPE_E']==diff_results['AIS-JCOPE_E']
    data = diff_results[tf]
    if len(data)==0:
        diff_ais_jcope_e = np.nan
    else:
        diff_ais_jcope_e = np.sqrt(np.sum(data['AIS-JCOPE_E']*data['AIS-JCOPE_E'])/len(data))
    data_num_ais_jcope = len(data)

    tf = diff_results['AIS-Ship_N']==diff_results['AIS-Ship_N']
    data = diff_results[tf]
    if len(data)==0:
        diff_ais_ship_n = np.nan
    else:
        diff_ais_ship_n = np.sqrt(np.sum(data['AIS-Ship_N']*data['AIS-Ship_N'])/len(data))
    
    tf = diff_results['AIS-Ship_E']==diff_results['AIS-Ship_E']
    data = diff_results[tf]
    if len(data)==0:
        diff_ais_ship_e = np.nan
    else:
        diff_ais_ship_e = np.sqrt(np.sum(data['AIS-Ship_E']*data['AIS-Ship_E'])/len(data))
    data_num_ais_ship = len(data)
    
    tf = diff_results['JCOPE-Ship_N']==diff_results['JCOPE-Ship_N']
    data = diff_results[tf]
    if len(data)==0:
        diff_jcope_ship_n = np.nan
    else:
        diff_jcope_ship_n = np.sqrt(np.sum(data['JCOPE-Ship_N']*data['JCOPE-Ship_N'])/len(data))
    tf = diff_results['JCOPE-Ship_E']==diff_results['JCOPE-Ship_E']
    data = diff_results[tf]
    if len(data)==0:
        diff_jcope_ship_e = np.nan
    else:
        diff_jcope_ship_e = np.sqrt(np.sum(data['JCOPE-Ship_E']*data['JCOPE-Ship_E'])/len(data))
    data_num_jcope_ship = len(data)

    print('-------------------------')
    print('curN, curE, num_data')
    print(f'AIS-JCOPE: {diff_ais_jcope_n:.2f}, {diff_ais_jcope_e:.2f}, {data_num_ais_jcope}')
    print(f'AIS-Ship: {diff_ais_ship_n:.2f}, {diff_ais_ship_e:.2f}, {data_num_ais_ship}')
    print(f'JCOPE-Ship: {diff_jcope_ship_n:.2f}, {diff_jcope_ship_e:.2f}, {data_num_jcope_ship}')
    print('-------------------------')
    print('\n')
    return [diff_ais_jcope_n, diff_ais_jcope_e, data_num_ais_jcope],\
            [diff_ais_ship_n, diff_ais_ship_e, data_num_ais_ship],\
            [diff_jcope_ship_n, diff_jcope_ship_e, data_num_jcope_ship]

# JCOPE (MAP curN, curE)
jcope_n_path = {}
jcope_e_path = {}
for day in range(1, n_day+1):
    for hour in range(24):
        dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
        dtidx = date_to_dtidx(base_dt, dt)
        jcope_n_path[dtidx] = osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "N.csv")
        jcope_e_path[dtidx] = osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "E.csv")
        # jcope_n[dtidx] = pd.read_csv(osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "N.csv"), encoding="cp932", header=None)
        # jcope_n[dtidx] = jcope_n[dtidx].values
        # jcope_e[dtidx] = pd.read_csv(osp.join(path_jcope, f'{year}{month:02}{day:02}{hour:02}', "E.csv"), encoding="cp932", header=None)
        # jcope_e[dtidx] = jcope_e[dtidx].values

ais_n_path = {}
ais_e_path = {}
ais_d_path = {}
patterns = [f'(\w+){month:02}..[0-9][0-9]N.csv', 
            f'(\w+){month:02}..[0-9][0-9]E.csv', 
            f'(\w+){month:02}..[0-9][0-9]D.csv']
files = os.listdir(path_ais)
dirs = [f for f in files if os.path.isdir(path_ais)]
for d in dirs:
    path2 = osp.join(path_ais, d, 'log')
    files = os.listdir(path2)
    filenames = [f for f in files if os.path.isfile(osp.join(path2, f))]
    for f in filenames:
        if re.match(patterns[0], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_n_path.keys():
                ais_n_path[dtidx] = []
                ais_n_path[dtidx].append(f_path)
                # ais_n_path[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_n_path[dtidx][-1] = ais_n_path[dtidx][-1].values
            else:
                ais_n_path[dtidx].append(f_path)
                # ais_n[dtidx].append(pd.read_csv(f_path, encoding="cp932", header=None))
                # ais_n[dtidx][-1] = ais_n[dtidx][-1].values         
        if re.match(patterns[1], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_e_path.keys():
                ais_e_path[dtidx] = []
                ais_e_path[dtidx].append(f_path)
            else:
                ais_e_path[dtidx].append(f_path)

        if re.match(patterns[2], f):
            f_path = osp.join(path2, f)
            day = int(f[-9:-7])
            hour = int(f[-7:-5])
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt) 
            if not dtidx in ais_d_path.keys():
                ais_d_path[dtidx] = []
                ais_d_path[dtidx].append(f_path)
            else:
                ais_d_path[dtidx].append(f_path)    

diff_ais_jcope1 = []
diff_ais_ship1 = []
diff_jcope_ship1 = []
diff_ship_name1 = []

diff_ais_jcope2 = []
diff_ais_ship2 = []
diff_jcope_ship2 = []
diff_ship_name2 = []

files = os.listdir(path)
target_ships = dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]

for day in range(1, n_day+1):
#for day in range(23, 23+1):
    dt = datetime.datetime(dt_year, dt_month, day, 0, 0, 0)
    print(dt)
    def read_ais(year, month, day):
        ais_n = {}
        ais_e = {}
        ais_d = {}

        for hour in range(24):
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)

            dtidx = date_to_dtidx(base_dt, dt)
            ais_n[dtidx] = []
            ais_e[dtidx] = []
            ais_d[dtidx] = []
            
            if dtidx in ais_n_path.keys():
                for i in range(len(ais_n_path[dtidx])):
                    try:
                        ais_n[dtidx].append(pd.read_csv(ais_n_path[dtidx][i], encoding="cp932", header=None))
                        ais_n[dtidx][-1] = ais_n[dtidx][-1].values
                    except:
                        print(f'Error load {ais_n_path[dtidx][i]}')
            
            if dtidx in ais_e_path.keys():
                for i in range(len(ais_e_path[dtidx])):
                    try:
                        ais_e[dtidx].append(pd.read_csv(ais_e_path[dtidx][i], encoding="cp932", header=None))
                        ais_e[dtidx][-1] = ais_e[dtidx][-1].values  
                    except:
                        print(f'Error load {ais_e_path[dtidx][i]}')            
            
            if dtidx in ais_d_path.keys():
                for i in range(len(ais_d_path[dtidx])):
                    try:
                        ais_d[dtidx].append(pd.read_csv(ais_d_path[dtidx][i], encoding="cp932", header=None))
                        ais_d[dtidx][-1] = ais_d[dtidx][-1].values  
                    except:
                        print(f'Error load {ais_d_path[dtidx][i]}')

        return ais_n, ais_e, ais_d

    def read_jcope(year, month, day):
        jcope_n = {}
        jcope_e = {}

        for hour in range(24):
            dt = datetime.datetime(dt_year, dt_month, day, hour, 0, 0)
            dtidx = date_to_dtidx(base_dt, dt)
            jcope_n[dtidx] = []
            jcope_e[dtidx] = []
            try:
                jcope_n[dtidx] = pd.read_csv(jcope_n_path[dtidx], encoding="cp932", header=None)
                jcope_n[dtidx] = jcope_n[dtidx].values
            except:
                print(f'Error load {jcope_n_path[dtidx]}')

            try:
                jcope_e[dtidx] = pd.read_csv(jcope_e_path[dtidx], encoding="cp932", header=None)
                jcope_e[dtidx] = jcope_e[dtidx].values
            except:
                print(f'Error load {jcope_e_path[dtidx]}')

        return jcope_n, jcope_e
    
    ais_n, ais_e, ais_d = read_ais(dt_year, dt_month, day)
    jcope_n, jcope_e = read_jcope(dt_year, dt_month, day)


    for target_ship in target_ships:
        print(f'{target_ship} {dt_month:02}{day:02}')
        if target_ship in done_ship:
            print(f'skip {target_ship}')
            continue
        # patterns = []
        # patterns.append(fr'cur_hours{dt_year}{dt_month:02}{day:02}.csv')
        # forbid_patterns = []

        # not_existData = True
        # path_name = []
        # path_logs = []
        # path2 = osp.join(path_ship, target_ship, '2015')
        # try:
        #     files = os.listdir(path2)
        # except:
        #     print(f'not found {path2}')
        # filenames = [f for f in files if os.path.isfile(osp.join(path2, f))]
        # for pattern in patterns:
        #     for f in filenames:
        #         if re.match(pattern, f):
        #             forbid = False
        #             for forbid_pattern in forbid_patterns:
        #                 if re.match(forbid_pattern, f):
        #                     forbid = True
        #                     break
        #             if not forbid:
        #                 f_path = osp.join(path2, f)
        #                 try:
        #                     log = pd.read_csv(f_path, encoding="cp932")
        #                     log['ShipName'] = target_ship
        #                     if not_existData:
        #                         shipLog = log
        #                         not_existData = False
        #                     else:
        #                         shipLog = pd.concat([shipLog, log], axis=0)
        #                     path_name.append(f_path)
        #                     path_logs.append(path2)
        #                     print(f_path)
        #                 except:
        #                     print(f'Error {f_path}')
        #         else:
        #             continue
        # if not_existData:
        #     print(f'not exist file in {target_ship}, day={day}')
        #     continue
        # elif len(shipLog)==0:
        #     print(f'not exist data in {target_ship}, day={day}')
        #     continue

        f_name = fr'cur_hours{dt_year}{dt_month:02}{day:02}.csv'
        f_path = osp.join(path_ship, target_ship, '2015', f_name)
        path_log = osp.join(path_ship, target_ship, '2015')
        try:
            shipLog = pd.read_csv(f_path, encoding="cp932")
            shipLog['ShipName'] = target_ship

            if len(shipLog)==0:
                print(f'not exist data in {target_ship}, day={day}')
                continue

        except:
            print(f'Error load {f_path}')
            continue

        print(f'ais n {len(ais_n)}, ais e {len(ais_e)}, ais d {len(ais_d)}')
        print(f'jcope n {len(jcope_n)}, jcope e {len(jcope_e)}')
        print(f'shiplog {len(shipLog)}')

        ais = [ais_n, ais_e]
        jcope = [jcope_n, jcope_e]

        diff_results = diff_ais_jcope_ship(ais, jcope, shipLog, use_d=False)
        save_path = osp.join(path_log, f"diff_results{dt_year}{dt_month:02}{day:02}.csv")
        diff_results.to_csv(save_path)
        diff_results['shipName'] = np.array([ target_ship for _ in range(len(diff_results))])
        diff_ais_jcope, diff_ais_ship, diff_jcope_ship = print_diff_ais_jcope_ship(diff_results)
        
        diff_ais_jcope1.append(diff_ais_jcope)
        diff_ais_ship1.append(diff_ais_ship)
        diff_jcope_ship1.append(diff_jcope_ship)
        diff_ship_name1.append(target_ship)

        diff_results = diff_ais_jcope_ship(ais, jcope, shipLog)
        save_path = osp.join(path_log, f"diff_results{dt_year}{dt_month:02}{day:02}_useAIS-D.csv")
        diff_results.to_csv(save_path)
        diff_ais_jcope, diff_ais_ship, diff_jcope_ship = print_diff_ais_jcope_ship(diff_results)

        diff_ais_jcope2.append(diff_ais_jcope)
        diff_ais_ship2.append(diff_ais_ship)
        diff_jcope_ship2.append(diff_jcope_ship)
        diff_ship_name2.append(target_ship)

diff_results = pd.DataFrame([])
a = np.array(diff_ais_jcope1)
diff_results['AIS-JCOPE_N'] = a.T[0] 
diff_results['AIS-JCOPE_E'] = a.T[1] 
diff_results['AIS-JCOPE_DataNum'] = a.T[2] 

a = np.array(diff_ais_ship1)
diff_results['AIS-Ship_N'] = a.T[0] 
diff_results['AIS-Ship_E'] = a.T[1] 
diff_results['AIS-Ship_DataNum'] = a.T[2] 

a = np.array(diff_jcope_ship1)
diff_results['JCOPE-Ship_N'] = a.T[0] 
diff_results['JCOPE-Ship_E'] = a.T[1] 
diff_results['JCOPE-Ship_DataNum'] = a.T[2] 

a = np.array(diff_ship_name1)
diff_results['shipName'] = a.T

save_path = osp.join(path_ship, f"diff_results{dt_year}{dt_month:02}.csv")
diff_results.to_csv(save_path, encoding='cp932')


diff_results = pd.DataFrame([])
a = np.array(diff_ais_jcope2)
diff_results['AIS-JCOPE_N'] = a.T[0] 
diff_results['AIS-JCOPE_E'] = a.T[1] 
diff_results['AIS-JCOPE_DataNum'] = a.T[2] 

a = np.array(diff_ais_ship2)
diff_results['AIS-Ship_N'] = a.T[0] 
diff_results['AIS-Ship_E'] = a.T[1] 
diff_results['AIS-Ship_DataNum'] = a.T[2] 

a = np.array(diff_jcope_ship2)
diff_results['JCOPE-Ship_N'] = a.T[0] 
diff_results['JCOPE-Ship_E'] = a.T[1] 
diff_results['JCOPE-Ship_DataNum'] = a.T[2] 

a = np.array(diff_ship_name2)
diff_results['shipName'] = a.T

save_path = osp.join(path_ship, f"diff_results{dt_year}{dt_month:02}_useAIS-D.csv")
diff_results.to_csv(save_path, encoding='cp932')
print(f'saved in {path_ship}')


    
