import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
from tqdm import tqdm
import pickle as pkl
import math
import os 
import os.path as osp
import re
import logtext

import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)

from utils import *
from kf_params import *

path = r"E:\shunsukeE\data\shiplog/"
files = os.listdir(path)
#dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]
"""target_ships  = {'中春': '中春丸',
                '2辰巳': '第二辰巳丸',
                '11和光': '第十一和光丸',
                '18英山': '第十八英山丸',
                '21東': '第二十一東丸',
                '33東洋': '第三十三東洋丸',
                '87東洋': '第八十七東洋丸',
                'MARS': 'SUNNYMARS',
                'ひま2': 'ひまわり２',
                '興春': '興春丸',
                '黒潮': '黒潮丸',
                '昇山': '昇山丸',
                '昭建': '昭建丸',
                '昭瑞': '昭瑞丸',
                '清栄': '清栄丸',
                '双信': '双信丸',
                '筑前': '筑前丸',
                '如月': '如月丸',
                '八菱': '第八菱洋丸',
                '豊鶴': '豊鶴丸',
                '立眞': '立眞丸'}"""
target_ships  = {'33東洋': '第三十三東洋丸'}

year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month)
done_ship = []

def latlon_to_mesh_df(lat, lon, deg_per_mesh=1/36, size=[1050, 1191], latlon_range=[20-1/36, 117-1/36]):
    # lat, lon -> [lon, lat]
    grid0 = ((lon-latlon_range[1]).astype(int)/deg_per_mesh)
    grid1 = ((lat-latlon_range[0]).astype(int)/deg_per_mesh)
    grid0[grid0 < 0] = -1
    grid0[grid0 > size[0]] = -1
    grid1[grid1 < 0] = -1
    grid1[grid1 > size[1]] = -1
    return grid0, grid1

def mean_ground_speed(dt, lat1, lon1, lat2, lon2):
    
    lat1 = dms_to_deg(lat1)
    lon1 = dms_to_deg(lon1)
    
    lat2 = dms_to_deg(lat2)
    lon2 = dms_to_deg(lon2)
    
    dist, deg = dist_deg_latlon(lat1, lon1, lat2, lon2)
    speed = dist/dt
    return ((lat1+lat2)/2, (lon1+lon2)/2), (speed, deg)

def dist_deg_latlon(lat1, lon1, lat2, lon2):
    from geopy.distance import geodesic
    res = geodesic((lat1, lon1), (lat2, lon2))
    
    # 方位角を計算
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    delta_lon = lon2_rad - lon1_rad
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(y, x))

    # 0から360度の範囲に調整
    bearing = (bearing + 360) % 360
    return res.meters, bearing

def dms_to_deg(x):
    deg = x // 100
    mit = x - deg*100
    #print(x)
    #print(deg)
    #print(mit)
    #print(sec)
    return deg + mit/60

def deg_to_rat(deg):
    #  北基準のdef
    return (-deg+90)*np.pi/180

def remove_outlier(df, key):
    # 下位・上位５％のデータを消す（外れ値対策）
    q1 = df[key].quantile(0.05)
    q2 = df[key].quantile(0.95)
    tf = (df[key]>q1) & (df[key]<q2)
    df2 = df[tf]
    return df2
    
def latlon_knot(lat1, lon1, lat2, lon2, deltaTime):
    # deltaTime = time_set[i+1]-time_set[i]
    distance = haversine_distance(lat1, lon1, lat2, lon2)/60
    knot = (distance/(deltaTime)) * 1.94384
    return knot

def divide_nmea(log_data):
    # 時間の整理　dtIdx: 0時からの経過時間，dtIdx_Minute:0時0分からの経過分
    time_utc =log_data["UTC"].values
    str_format = '%Y/%m/%d %H:%M:%S'
    epoc_dt = datetime.datetime(dt_year, dt_month, dt_day, 0, 0, 0)
    time_idx = []
    time_idx2 = []
    tf = []
    for t in time_utc:
        idx = datetime.datetime.strptime(t, str_format) 
        idx2 = idx - epoc_dt
        time_idx.append(int(idx2.days*24 + idx2.seconds / (60*60)))
        time_idx2.append(int(idx2.days*24*60 + idx2.seconds / (60)))
    time_idx = np.array(time_idx)
    time_idx2 = np.array(time_idx2)
    log_data["DtIdx"] = time_idx
    log_data["DtIdx_Minute"] = time_idx2

    # NMEAデータの抽出
    # ggaデータの抽出
    gga = log_data[log_data["NMEA"]=="GGA"]
    gga.columns = ['UTC', 'NMEA', 'LatDMS', 'LonDMS', 'DtIdx', 'DtIdx_Minute']
    gga['Lat'] = dms_to_deg(gga['LatDMS'])
    gga['Lon'] = dms_to_deg(gga['LonDMS'])

    # vbwデータの抽出
    vbw = log_data[log_data["NMEA"]=="VBW"]
    vbw.columns = ['UTC', 'NMEA', 'LonWaterSpeed', 'TraWaterSpeed', 'DtIdx', 'DtIdx_Minute']

    
    # hdtデータの抽出
    hdt = log_data[log_data["NMEA"]=="HDT"]
    hdt = hdt.drop('data2', axis=1)
    hdt.columns = ['UTC', 'NMEA', 'HeadDeg', 'DtIdx', 'DtIdx_Minute']

    # vtgデータの抽出    
    vtg = log_data[log_data["NMEA"]=="VTG"]
    vtg.columns = ['UTC', 'NMEA', 'HeadDeg', 'GroundSpeed', 'DtIdx', 'DtIdx_Minute']

    # vtgデータの抽出
    vhw = log_data[log_data["NMEA"]=="VHW"]
    vhw.columns = ['UTC', 'NMEA', 'HeadDeg', 'WaterSpeed', 'DtIdx', 'DtIdx_Minute']
    
    # データ処理
    delete_time_nmri = {}
    stop_knot = 0.5
    
    # vtgデータ処理
    prev_dt = set(vtg['DtIdx_Minute'].values)
    tf = vtg['GroundSpeed']>stop_knot
    vtg = vtg[tf]
    delete_time_nmri['VTG'] = prev_dt - set(vtg['DtIdx_Minute'].values)

    # vbwのデータ処理
    prev_dt = set(vbw['DtIdx_Minute'].values)
    tf = vbw['LonWaterSpeed']!=-999.0
    vbw = vbw[tf] 
    tf = vbw['LonWaterSpeed']>stop_knot
    vbw = vbw[tf] 
    delete_time_nmri['VBW'] = prev_dt - set(vbw['DtIdx_Minute'].values)
    
    # hdtデータ処理
    prev_dt = set(hdt['DtIdx_Minute'].values)
    time_set = sorted(set(hdt["DtIdx_Minute"]))
    delete_range = 10
    thres = (10/180)*np.pi
    for i in range(len(time_set)-1):
            hdt1 = hdt[time_set[i] == hdt["DtIdx_Minute"]]
            hdt2 = hdt[time_set[i+1] == hdt["DtIdx_Minute"]]
            
            hdt1 = remove_outlier(hdt1, 'HeadDeg')
            hdt2 = remove_outlier(hdt2, 'HeadDeg')
            
            headRat1 = deg_to_rat(hdt1['HeadDeg'])
            sin1 = np.mean(np.sin(headRat1))
            cos1 = np.mean(np.cos(headRat1))
            
            headRat2 = deg_to_rat(hdt2['HeadDeg'])
            sin2 = np.mean(np.sin(headRat2))
            cos2 = np.mean(np.cos(headRat2))

            delta_theta = np.arccos(sin1*sin2 + cos1*cos2)
            omega = delta_theta/(time_set[i+1]-time_set[i])
            
            time = time_set[i]
            if np.abs(omega)>thres:
                tf = hdt['DtIdx_Minute']!=time
                hdt = hdt[tf]
                for j in range(1, 10):
                    tf = hdt['DtIdx_Minute']!=time-j
                    hdt = hdt[tf]
                    tf = hdt['DtIdx_Minute']!=time+j
                    hdt = hdt[tf]
    delete_time_nmri['HDT'] = prev_dt - set(hdt['DtIdx_Minute'].values)
    # vhwデータ処理
    prev_dt = set(vhw['DtIdx_Minute'].values)
    tf = vhw['WaterSpeed']!=-999.0
    vhw = vhw[tf]
    tf = vhw['WaterSpeed']>stop_knot
    vhw = vhw[tf]
    delete_time_nmri['VHW'] = prev_dt - set(vhw['DtIdx_Minute'].values)
    
    return gga, vbw, hdt, vtg, vhw, delete_time_nmri

def cur_nmea(gga, vbw, hdt, vtg):
    min_timeHours = np.min(gga["DtIdx"])
    max_timeHours = np.max(gga["DtIdx"])
    num_timeHours = max_timeHours - min_timeHours + 1

    grids0 = []
    grids1 = []
    curN = []
    curE = []
    timeMinutes = []
    timeHours = []
    UTC_time = []
    lats = []
    lons = []
    curN_grid = {}
    curE_grid = {}
    count_grid = {}

    time_set_h = set(gga["DtIdx"])
    for s in time_set_h:
        curN_grid[s] = np.zeros(nan_map.shape)
        curE_grid[s] = np.zeros(nan_map.shape)
        count_grid[s] = np.zeros(nan_map.shape) 

    time_set = set(gga["DtIdx_Minute"])
    time_set_vtg = set(vtg["DtIdx_Minute"])
    print(time_set)
    print(time_set_vtg)
    input()
    for time in time_set:
        print('--------------------')
        gga1 = gga[time == gga["DtIdx_Minute"]]
        vbw1 = vbw[time == vbw["DtIdx_Minute"]]
        hdt1 = hdt[time == hdt["DtIdx_Minute"]]
        vtg1 = vtg[time == vtg["DtIdx_Minute"]]
        if (len(vbw1)==0 or len(gga1)==0 or len(hdt1)==0 or len(vtg1)==0): 
            if len(vbw1)==0:
                print('nodata vbw')
                logtext.add_text('not exist vbw')
            if len(gga1)==0:
                print('nodata gga') 
                logtext.add_text('not exist gga')
            if len(hdt1)==0:
                print('nodata hdt') 
                logtext.add_text('not exist hdt')
            if len(vtg1)==0:
                print('nodata vtg') 
                logtext.add_text('not exist vtg')
            continue

        # GGAからLat Lon 取得
        lat1 = np.mean(remove_outlier(gga1, 'Lat')['Lat'])
        lon1 = np.mean(remove_outlier(gga1, 'Lon')['Lon'])
        # データ数が少なすぎると、外れ値消去の際にnanになる，データが少ないときはその時間は計算せずcontinue
        if not lat1==lat1 or not lon1==lon1:
            print('not enought gga data')
            continue

        # HDTから船首方位取得
        headRat = deg_to_rat(hdt1['HeadDeg'])
        sin1 = np.mean(np.sin(headRat))
        cos1 = np.mean(np.cos(headRat))
        

        # VBWから対船水速取得
        water_speed = np.mean(remove_outlier(vbw1, 'LonWaterSpeed')['LonWaterSpeed'])
        if not water_speed==water_speed:
            print('not enought vbw data')
            logtext.add_text('not enought vbw')
            continue

        # VTGから船首方位・対地船速取得
        g_headRat = deg_to_rat(vtg1['HeadDeg'])
        g_sin1 = np.mean(np.sin(g_headRat))
        g_cos1 = np.mean(np.cos(g_headRat))
        ground_speed = np.mean(remove_outlier(vtg1, 'GroundSpeed')['GroundSpeed'])
        if not ground_speed==ground_speed:
            print('not enought vtg data')
            logtext.add_text('not enought vtg')
            continue

        # 偏流の計算
        curN.append(ground_speed*g_sin1 - water_speed*sin1)
        curE.append(ground_speed*g_cos1 - water_speed*cos1)
        #curN.append(-water_speed*sin1 + ground_speed*g_sin1)
        #curE.append(-water_speed*cos1 + ground_speed*g_cos1)
        print(f'N {ground_speed*g_sin1 - water_speed*sin1}')
        print(f'E {ground_speed*g_cos1 - water_speed*cos1}')
        
        # Gridの位置計算
        grid0, grid1 = latlon_to_mesh(lat1, lon1)

        grids0.append(grid0)
        grids1.append(grid1)
        lats.append(lat1)
        lons.append(lon1)

        UTC_time.append(gga1['UTC'].values[-1][:-6])
        timeMinutes.append(time)
        timeHours.append(gga1['DtIdx'].values[-1])
        curN_grid[timeHours[-1]][grid0][grid1] += curN[-1]
        curE_grid[timeHours[-1]][grid0][grid1] += curE[-1]
        count_grid[timeHours[-1]][grid0][grid1] += 1
        print('--------------------')

    grid_cur_m = pd.DataFrame([])
    grid_cur_m["DtIdx"] = timeHours
    grid_cur_m["DtIdx_Minute"] = timeMinutes
    grid_cur_m["UTC"] = UTC_time
    grid_cur_m["CurN"] = curN
    grid_cur_m["CurE"] = curE
    grid_cur_m["Grid0"] = grids0
    grid_cur_m["Grid1"] = grids1
    grid_cur_m["Lat"] = lats
    grid_cur_m["Lon"] = lons
    for i in time_set_h:
        count_grid[i][count_grid[i] == 0] += 1
        curN_grid[i] = curN_grid[i]/count_grid[i]
        curE_grid[i] = curE_grid[i]/count_grid[i]
    return grid_cur_m, curN_grid, curE_grid

    for i in time_set_h:
        count_grid[i][count_grid[i] == 0] += 1
        curN_grid[i] = curN_grid[i]/count_grid[i]
        curE_grid[i] = curE_grid[i]/count_grid[i]
    return grid_cur_m, curN_grid, curE_grid

    # vhwデータ処理
    prev_dt = set(vhw['DtIdx_Minute'].values)
    tf = vhw['WaterSpeed']!=-999.0
    vhw = vhw[tf]
    tf = vhw['WaterSpeed']>stop_knot
    vhw = vhw[tf]
    delete_time_nmri['VHW'] = prev_dt - set(vhw['DtIdx_Minute'].values)
    
    return gga, vbw, hdt, vtg, vhw, delete_time_nmri

def cur_minute_to_hour(grid_cur_m):
    grids0 = []
    grids1 = []
    curN = []
    curE = []
    timeHours = []
    UTC_time = []
    lats = []
    lons = []

    time_set = set(grid_cur_m["DtIdx"])
    for time in time_set:
        target = grid_cur_m[time == grid_cur_m["DtIdx"]]
        timeHours.append(time)
        UTC_time.append(target["UTC"].values[0])
        lats.append(np.mean(target["Lat"].values))
        lons.append(np.mean(target["Lon"].values))
        grids0.append(int(np.mean(target["Grid0"].values)))
        grids1.append(int(np.mean(target["Grid1"].values)))
        curN.append(np.mean(target["CurN"].values))
        curE.append( np.mean(target["CurE"].values))
#         lats.append(target.Lat.quantile(0.95))
#         lons.append(target.Lon.quantile(0.95))
#         grids0.append(int(target.Grid0.quantile(0.95)))
#         grids1.append(int(target.Grid1.quantile(0.95)))
#         curN.append(target.CurN.quantile(0.95))
#         curE.append(target.CurE.quantile(0.95))
    grid_cur = pd.DataFrame([])
    grid_cur["DtIdx"] = timeHours
    grid_cur["UTC"] = UTC_time
    grid_cur["CurN"] = curN
    grid_cur["CurE"] = curE
    grid_cur["Grid0"] = grids0
    grid_cur["Grid1"] = grids1
    grid_cur["Lat"] = lats
    grid_cur["Lon"] = lons
    return grid_cur

target_days = [18]
for target_ship in target_ships.keys():
    logtext.clear()
    logtext.add_text(f'{target_ship}')
    print(f'{target_ship}')
    if target_ship in done_ship:
        print(f'skip {target_ship}')
        continue
    patterns = []
    #for day in range(1, n_day+1):
    for day in target_days:
        patterns.append(fr'(\w+){month:02}{day:02}.slog1')
    forbid_patterns = [fr'(\w+).slog1err']

    log_datas = []
    path_name = []
    path_logs = []
    path2 = osp.join(path, target_ship, '2015')
    try:
        files = os.listdir(path2)
    except:
        print(f'not found {path2}')
        logtext.add_text(f'not found {path2}')
    filenames = [f for f in files if os.path.isfile(osp.join(path2, f))]
    for pattern in patterns:
        for f in filenames:
            if re.match(pattern, f):
                forbid = False
                for forbid_pattern in forbid_patterns:
                    if re.match(forbid_pattern, f):
                        forbid = True
                        break
                if not forbid:
                    f_path = osp.join(path2, f)
                    try:
                        log = pd.read_csv(f_path, encoding="cp932", header=None)
                        log.columns = ["UTC", "NMEA", "data1", "data2"]
                        log_datas.append(log)
                        path_name.append(f_path)
                        path_logs.append(path2)
                        print(f_path)
                    except:
                        print(f'Error {f_path}')
                        logtext.add_text(f'cannot load {f_path}')

            else:
                continue
    print(f'loaded file num: {len(log_datas)}')
    for i in range(len(log_datas)):
        log_data = log_datas[i]
        print(f'--------num data--------')
        print(f"FileName: {path_name[i]}")
        print(f'GGA: {len(log_data[log_data["NMEA"]=="GGA"])}')
        print(f'VBW: {len(log_data[log_data["NMEA"]=="VBW"])}')
        print(f'HDT: {len(log_data[log_data["NMEA"]=="HDT"])}')
        print(f'VTG: {len(log_data[log_data["NMEA"]=="VTG"])}')
        print(f'VHW: {len(log_data[log_data["NMEA"]=="VHW"])}')
        print(f'------------------------\n')

    ggas = []
    vbws = []
    hdts = []
    vtgs = []
    vhws = []
    delete_times = []

    for i in range(len(log_datas)):
        gga, vbw, hdt, vtg, vhw,  delete_time_nmri = divide_nmea(log_datas[i])
        ggas.append(gga)
        vbws.append(vbw)
        hdts.append(hdt)
        vtgs.append(vtg)
        vhws.append(vhw)
        delete_times.append(delete_time_nmri)

    for i in range(len(log_datas)):
        log_data = log_datas[i]
        num_data = f"""FileName: {path_name[i]}\n
                       GGA: {len(ggas[i])}\n
                       VBW: {len(vbws[i])}\n
                       HDT: {len(hdts[i])}\n
                       VTG: {len(vtgs[i])}\n
                       VHW: {len(vhws[i])}\n"""
        logtext.add_text(num_data)
        print(f'--------num data--------')
        print(f'{num_data}')
        print(f'------------------------\n')


    grid_cur_minute = []
    curN_grid = []
    curE_grid = []
    for i in range(len(ggas)):
        grid_cur, N, E = cur_nmea(ggas[i], vbws[i], hdts[i], vtgs[i])
        grid_cur_minute.append(grid_cur)
        curN_grid.append(N)
        curE_grid.append(E)
        
    grid_curs = []
    for grid_cur_m in grid_cur_minute:
        grid_cur = cur_minute_to_hour(grid_cur_m)
        grid_curs.append(grid_cur)

    for i in range(len(grid_curs)):
        # 保存
        gird_cur_m = grid_cur_minute[i]
        grid_cur = grid_curs[i]
        grid_cur_m = grid_cur_m.sort_values('DtIdx_Minute')
        grid_cur_m = grid_cur_m.reset_index(drop=True)
        grid_cur = grid_cur.sort_values('DtIdx')
        grid_cur = grid_cur.reset_index(drop=True)

        day = path_name[i][-8:-6]
        save_path = osp.join(path_logs[i], f"cur_hours{dt_year}{dt_month:02}{day}.csv")
        grid_cur.to_csv(save_path)

        save_path = osp.join(path_logs[i], f"cur_minutes{dt_year}{dt_month:02}{day}.csv")
        grid_cur_m.to_csv(save_path)
        
        pkl.dump(curN_grid[i], open(osp.join(path_logs[i], f"curGridN.pkl"), 'wb'))
        pkl.dump(curE_grid[i], open(osp.join(path_logs[i], f"curGridE.pkl"), 'wb'))
        print(save_path)
    fname = osp.join(path2, 'making_cur_log.txt')
    logtext.output(fname)
    #except:
    #    print(f'Error {target_ship}')