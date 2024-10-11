import argparse
import math
import csv
import numpy as np
import pickle as pkl
import math
import os 
import os.path as osp
import pandas as pd

from utils import *
from utils.utils_needed_params import *

#他のライブラリ内でimport datetimeがあるため最後にimportしとく
from datetime import datetime, timedelta 

deg_per_mesh = 1/(36/pool_size)
nan_map_path = fr'logs/nan_map_pooled{pool_size}.csv'
df_nan_map_pooled = pd.DataFrame(nan_map_pooled)
df_nan_map_pooled.to_csv(nan_map_path, index=False, header=False)

class KurosioRangePluse():
    def __init__(self, deltax, deltay):
        line1 = [[kurosio_lon_range[0], kurosio_lat_range1[0]], [kurosio_lon_range[1], kurosio_lat_range2[0]]]
        a1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
        b1 = line1[0][1] - deltay - a1 * line1[0][0]
        y1 = a1 * (line1[0][0] - deltax) + b1
        y2 = a1 * (line1[1][0] + deltax) + b1
        new_line1 = [[kurosio_lon_range[0]-deltax, y1], [kurosio_lon_range[1]+deltax, y2]]
        self.under_line = new_line1
        self.a1 = a1
        self.b1 = b1

        line2 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]
        a2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
        b2 = line2[0][1] + deltay - a2 * line2[0][0]
        y1 = a2 * (line2[0][0] - deltax) + b2
        y2 = a2 * (line2[1][0] + deltax) + b2
        new_line2 = [[kurosio_lon_range[0]-deltax, y1], [kurosio_lon_range[1]+deltax, y2]]
        self.upper_line = new_line2
        self.a2 = a2
        self.b2 = b2
 
    def is_in_kurosio(self, lat, lon):
        if lon<self.under_line[0][0] or lon>self.under_line[1][0]:
            return False
        if lat<self.under_line[0][1] or lat>self.upper_line[1][1]:
            return False
        else:
            y1 = self.a1 * lon + self.b1
            y2 = self.a2 * lon + self.b2
            if lat<y1:
                return False
            if lat>y2:
                return False
        return True

    
    def print_info(self):
        print(f'under line: {self.under_line}')
        print(f'y = {self.a1} x + {self.b1}')
        print(f'upper line: {self.upper_line}')
        print(f'y = {self.a2} x + {self.b2}')

    def is_in_kurosio_test(self, lat, lon):
        print(f'Checked (lat, lon)=({lat}, {lon})')
        if lon<self.under_line[0][0] or lon>self.under_line[1][0]:
            print('out of lon range')
            return False
        if lat<self.under_line[0][1] or lat>self.upper_line[1][1]:
            print('out of lat range')
            return False
        else:
            y1 = self.a1 * lon + self.b1
            y2 = self.a2 * lon + self.b2
            if lat<y1:
                print(f'under lat range y1={y1} (but in square area)')
                return False
            if lat>y2:
                print(f'over lat range y2={y2} (but in square area)')
                return False
        print('Inside of kurosio')
        return True
class Settei:
    # latidx = latdeg * 36 / pool_size
    latlon_maxmin = {'LatIdxMax1': 1800,
                     'LatIdxMin1': 750,
                     'LonIdxMax1': 5401,
                     'LonIdxMin1': 4211,
                     'LatIdxMax2': 900,
                     'LatIdxMin2': 376,
                     'LonIdxMax2': 2700,
                     'LonIdxMin2': 2106,
                     'LatIdxMax3': 600,
                     'LatIdxMin3': 251,
                     'LonIdxMax3': 1800,
                     'LonIdxMin3': 1404,
                     'LatIdxMax6': 300,
                     'LatIdxMin6': 126,
                     'LonIdxMax6': 899,
                     'LonIdxMin6': 702,
                     }
    LatIdxMax = latlon_maxmin[f'LatIdxMax{pool_size}'] 
    LatIdxMin = latlon_maxmin[f'LatIdxMin{pool_size}']
    LonIdxMax = latlon_maxmin[f'LonIdxMax{pool_size}']
    LonIdxMin = latlon_maxmin[f'LonIdxMin{pool_size}']

    DtIdxMax = 0
    DtIdxMin = 0
    tmpDtIdxMin = 0
    OutFolder = ""
    HalfLat = 0.1
    HalfLon = 0.2
    HalfHrs = 0.1
    ln2 =  0.6931471805599453094172321
    Thres = 0.05
    degPerMesh = deg_per_mesh
    hrsPerMesh = 1.0
    LatRange = int(0) #int(-HalfLat * math.log(Thres) / ln2 / degPerMesh)
    LonRange = int(0) #int(-HalfLon * math.log(Thres) / ln2 / degPerMesh)
    krp = KurosioRangePluse(-HalfLon * math.log(Thres) / ln2, -HalfLat * math.log(Thres) / ln2)
    #HrsRange = int(-HalfHrs * math.log(Thres) / ln2 / hrsPerMesh)
    HrsRange = int(0)
    h = HrsRange * 2 + 1
    weights = None
    Map = None
    NanMap = None
    LatLonIsValid = None

    @staticmethod
    def weight(DLat, DLon, DTime):
        DLat = abs(DLat)
        DLon = abs(DLon)
        DTime = abs(DTime)
        if DLat > Settei.LatRange or DLon > Settei.LonRange or DTime > Settei.HrsRange:
            return 0
        return Settei.weights[DTime][DLat][DLon]

    @staticmethod
    def init(dtIdxMax, dtIdxMin, outFolder):
        Settei.DtIdxMax = dtIdxMax
        Settei.DtIdxMin = dtIdxMin
        Settei.tmpDtIdxMin = Settei.DtIdxMin
        Settei.OutFolder = outFolder

        Settei.Map = [[[[
            0.0 for _ in range(5)
        ] for _ in range(Settei.LonIdxMin, Settei.LonIdxMax + 1)] for _ in range(Settei.LatIdxMin, Settei.LatIdxMax + 1)] for _ in range(Settei.h)]

        Settei.NanMap = [
            [ 0.0 for _ in range(Settei.LonIdxMin, Settei.LonIdxMax + 1)] for _ in range(Settei.LatIdxMin, Settei.LatIdxMax + 1)]

        Settei.LatLonIsValid = [
            [False for _ in range(Settei.LonIdxMin, Settei.LonIdxMax + 1)] for _ in range(Settei.LatIdxMin, Settei.LatIdxMax + 1)]

        with open(nan_map_path, "r") as sr:
            for i, line in enumerate(sr.readlines()):
                ss = line.split(',')
                k = Settei.LatIdxMax - Settei.LatIdxMin - i
                if k < 0:
                    break

                for j, value in enumerate(ss):
                    if j > Settei.LonIdxMax - Settei.LonIdxMin:
                        break
                    Settei.LatLonIsValid[k][j] = (value.strip() != "")

        Settei.weights = [
            [
                [math.exp(-Settei.ln2 * math.sqrt(
                    (lat * Settei.degPerMesh / Settei.HalfLat)**2 +
                    (lon * Settei.degPerMesh / Settei.HalfLon)**2 +
                    (hr * Settei.hrsPerMesh / Settei.HalfHrs)**2
                )) for lon in range(Settei.LonRange + 1)] for lat in range(Settei.LatRange + 1)
            ] for hr in range(Settei.HrsRange + 1)
        ]

    @staticmethod
    def add_map(DtIdx, LatIdx, LonIdx, item, Value):
        if DtIdx < Settei.DtIdxMin or DtIdx > Settei.DtIdxMax:
            return
        if DtIdx < Settei.tmpDtIdxMin:
            print(f"Error DtIdx={DtIdx} < tmpDtIdxMin={Settei.tmpDtIdxMin}")
            return
        # 新しい時間帯（dtidx）に移ったら今までの時間帯の偏流を計算
        while Settei.tmpDtIdxMin + Settei.h <= DtIdx:
            Settei.map_to_csv_and_clear(Settei.tmpDtIdxMin)
            Settei.tmpDtIdxMin += 1

        Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][item] += Value

    @staticmethod
    def add_nanmap(LatIdx, LonIdx):
        Settei.NanMap[LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin] = 1 


    @staticmethod
    def close():
        for DtIdx in range(Settei.tmpDtIdxMin, Settei.DtIdxMax + 1):
            #print(f'close dtidx {DtIdx}')
            Settei.map_to_csv_and_clear(DtIdx)

    @staticmethod
    def is_lat_lon_valid(LatIdx, LonIdx):
        return Settei.LatLonIsValid[LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin]

    @staticmethod
    def map_to_csv_and_clear(DtIdx):
        dt_format = datetime(2011, 1, 1) + timedelta(hours=DtIdx)
        # print(f"Output : {dt_format.strftime('%Y%m%d%H')}")
        # 注意！ X.csv: North, Y.csv: East
        with open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Cur1.csv", "w") as sw_cur1, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Cur2.csv", "w") as sw_cur2, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}X.csv", "w") as sw_x, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Y.csv", "w") as sw_y, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}D.csv", "w") as sw_d, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Lambda1.csv", "w") as sw_lambda1, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Lambda2.csv", "w") as sw_lambda2, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Phi1.csv", "w") as sw_phi1, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Phi2.csv", "w") as sw_phi2:

            for LatIdx in range(Settei.LatIdxMax, Settei.LatIdxMin - 1, -1):
                s_cur1 = ""
                s_cur2 = ""
                s_x = ""
                s_y = ""
                s_d = ""
                s_lambda1 = ""
                s_lambda2 = ""
                s_phi1 = ""
                s_phi2 = ""

                for LonIdx in range(Settei.LonIdxMin, Settei.LonIdxMax + 1):
                    A11 = Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][0]
                    A12 = Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][1]
                    A22 = Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][2]
                    B1 = Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][3]
                    B2 = Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][4]
                    D = A11 * A22 - A12 * A12

                    if D >= 1:
                        Lambda1 = (A11 + A22 - (math.sqrt((A11 - A22)**2 + 4 * A12**2))) / 2
                        Lambda2 = (A11 + A22 + (math.sqrt((A11 - A22)**2 + 4 * A12**2))) / 2
                        Phi1 = math.atan2(Lambda1 - A11, A12)
                        Phi2 = math.atan2(Lambda2 - A11, A12)
                        #X = B1/Lambda1
                        #Y = B2/Lambda2
                        #Cur1 = (X * math.cos(Phi1) + Y * math.sin(Phi1))
                        #Cur2 = (X * math.cos(Phi2) + Y * math.sin(Phi2))
                        X =  (A22 * B1 - A12 * B2) / D
                        Y =  (-A12 * B1 + A11 * B2) / D
                        Cur1 = (B1 * math.cos(Phi1) + B2 * math.sin(Phi1))/Lambda1
                        Cur2 = (B1 * math.cos(Phi2) + B2 * math.sin(Phi2))/Lambda2
                        s_cur1 += f"{Cur1:.2f},"
                        s_cur2 += f"{Cur2:.2f},"
                        s_x += f"{X:.2f},"
                        s_y += f"{Y:.2f},"
                        s_d += f"{D:.2f},"
                        s_lambda1 += f"{Lambda1:.2f},"
                        s_lambda2 += f"{Lambda2:.2f},"
                        s_phi1 += f"{Phi1:.2f},"
                        s_phi2 += f"{Phi2:.2f},"
                    else:
                        s_cur1 += f","
                        s_cur2 += f","
                        s_x += f","
                        s_y += f","
                        s_d += f","
                        s_lambda1 += f","
                        s_lambda2 += f","
                        s_phi1 += f","
                        s_phi2 += f","

                sw_cur1.write(s_cur1[:-1] + "\n")
                sw_cur2.write(s_cur2[:-1] + "\n")
                sw_x.write(s_x[:-1] + "\n")
                sw_y.write(s_y[:-1] + "\n")
                sw_d.write(s_d[:-1] + "\n")
                sw_lambda1.write(s_lambda1[:-1] + "\n")
                sw_lambda2.write(s_lambda2[:-1] + "\n")
                sw_phi1.write(s_phi1[:-1] + "\n")
                sw_phi2.write(s_phi2[:-1] + "\n")
        for LatIdx in range(Settei.LatIdxMax, Settei.LatIdxMin - 1, -1):
            for LonIdx in range(Settei.LonIdxMin, Settei.LonIdxMax + 1):
                for i in range(5):
                    Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][i] = 0
class Ais4ToCur:
    inExt = "ais4"
    outExt = "csv"
    logFile = "logout.log"
    startTime = datetime.max

    def __init__(self):
        self.ais4s = None
        self.outFolder = "data/ais"

    def set_outFolder(self, outFolder):
        try:
            if not os.path.exists(outFolder):
                os.makedirs(outFolder)
        except:
            print(f'Error mkdir {outFolder}')
        self.outFolder = outFolder
            
    def load_ais4(self, inFiles, target_dtidx):
            
        dtMin = float('inf')
        dtMax = float('-inf')
        ais4s = []
        Ais4ToCur.logout("Start Reading")

        ls = 0
        map_ndata = {}
        for inFile in inFiles:
            with open(inFile, "r") as sr:
                sr.readline()  # Skip first line
                sr.readline()  # Skip header
                l = 0
                ignore_dt = 0
                ignore_area = 0
                while True:
                    line = sr.readline()
                    if not line:
                        break
                    #print("{:.2f}%,{}lines done\r".format(100.0 * sr.tell() / os.path.getsize(inFile), l), end="")
                    l += 1
                    ss = line.split(',')

                    MMSI = int(ss[0])
                    DtIdx = int(ss[1])
                    dtMin = min(dtMin, DtIdx)
                    dtMax = max(dtMax, DtIdx)
                    
                    # ターゲットの時間以外は無視
                    if DtIdx!=target_dtidx:
                        ignore_dt += 1 
                        continue

                    LatIdx = int(int(ss[2])/(36*deg_per_mesh))
                    LonIdx = int(int(ss[3])/(36*deg_per_mesh))
                    
                    # 黒潮領域外は無視
                    if not Settei.krp.is_in_kurosio(LatIdx*deg_per_mesh, LonIdx*deg_per_mesh):
                        ignore_area += 1 
                        continue
                    
                    ThetaDeg = float(ss[4])
                    F = float(ss[5])
                    LineCount = float(ss[6])
                    ais4s.append(ais4(MMSI, DtIdx, LatIdx, LonIdx, ThetaDeg, F, LineCount))
                    key = f'({LatIdx},{LonIdx})'
                    if key in map_ndata:
                        map_ndata[key] += 1
                    else:
                        map_ndata[key] = 1
            print(f'lines: {l}, ignore dt: {ignore_dt}, ignore area: {ignore_area}')
            ls += l

        print(f'grid n data:{len(map_ndata.keys())}')
        ais4s.sort(key=lambda x: (x.DtIdx, x.LatIdx, x.LonIdx, x.MMSI))
        self.ais4s = ais4s
        if len(ais4s)==0:
            print(f'There is no data in ais4s. TargetDtIDx:{target_dtidx}')
            print(f'inFiles:{inFiles}')
        Settei.init(target_dtidx, target_dtidx, self.outFolder)
        #print(f'MinDtIdx: {dtMin}, MaxDtIdx:{dtMax}, TargetDtIdx:{target_dtidx}')

    def load_ais4s(self, filepath, dt):
        ais4s = pkl.load(open(f"{filepath}/Ais4s{dt.strftime('%Y%m%d%H')}.pkl", 'rb'))
        self.ais4s = ais4s
        if len(ais4s)==0:
            print(f'There is no data in ais4s. TargetDtIDx:{target_dtidx}')
        
        target_dtidx = date_to_dtidx(datetime(2011, 1, 1), dt)
        Settei.init(target_dtidx, target_dtidx, self.outFolder)
        
    def dump_ais4(self, dt, pkl_path):
        filename = f"{pkl_path}/Ais4s{dt.strftime('%Y%m%d%H')}.pkl"
        pkl.dump(self.ais4s, open(filename, 'wb'))
        print(f'Saved {filename}, Num ais4s: {len(self.ais4s)}')
        
    def save_nanmap(self):
        for latidx in range(Settei.LatIdxMin, Settei.LatIdxMax + 1):
            for lonidx in range(Settei.LonIdxMin, Settei.LonIdxMax + 1):
                if Settei.is_lat_lon_valid(latidx, lonidx):
                    Settei.add_nanmap(latidx, lonidx)

        with open(f"{Settei.OutFolder}/AisNanmap.csv", "w") as sw_nanmap:
            for LatIdx in range(Settei.LatIdxMax, Settei.LatIdxMin - 1, -1):
                s_nanmap = ""

                for LonIdx in range(Settei.LonIdxMin, Settei.LonIdxMax + 1):
                    value = Settei.NanMap[LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin]
                    if value==1:
                        s_nanmap += f"{value:.2f},"
                    else:
                        s_nanmap += f","

                sw_nanmap.write(s_nanmap[:-1] + "\n")

    def calc_cur(self, target_dtidx):
        ais4s = self.ais4s
        curs = []
        for i, a in enumerate(ais4s):
            #print("{:.2f}%,{}lines done\r".format(100.0 * i / len(ais4s), i), end="")
            if a.DtIdx!=target_dtidx:
                continue
            #print(f'dt{target_dtidx}, DT{a.DtIdx}')

            theta = a.ThetaDeg * (math.pi / 180)
            sin = math.sin(theta)
            cos = math.cos(theta)
            coscos = cos * cos
            sincos = sin * cos
            sinsin = sin * sin
            cosF = cos * a.Flow
            sinF = sin * a.Flow
            curN = a.Flow * cos
            curE = a.Flow * sin
            dtidx = a.DtIdx
            curs.append(Cur(a.MMSI, a.DtIdx, a.LatIdx, a.LonIdx, curN, curE))

            for latidx in range(a.LatIdx - Settei.LatRange, a.LatIdx + Settei.LatRange + 1):
                for lonidx in range(a.LonIdx - Settei.LonRange, a.LonIdx + Settei.LonRange + 1):
                    if Settei.LatIdxMin <= latidx <= Settei.LatIdxMax and \
                            Settei.LonIdxMin <= lonidx <= Settei.LonIdxMax and \
                            Settei.DtIdxMin <= dtidx <= Settei.DtIdxMax and \
                            Settei.is_lat_lon_valid(latidx, lonidx):
                        b = a.LineCount / (1 + 2 * a.LineCount * svm.get_var(a.MMSI))
                        w = Settei.weight(latidx - a.LatIdx, lonidx - a.LonIdx, dtidx - a.DtIdx) * b   
                        # 偏流に使う値を追加                     
                        Settei.add_map(dtidx, latidx, lonidx, 0, w * coscos)
                        Settei.add_map(dtidx, latidx, lonidx, 1, w * sincos)
                        Settei.add_map(dtidx, latidx, lonidx, 2, w * sinsin)
                        Settei.add_map(dtidx, latidx, lonidx, 3, w * cosF)
                        Settei.add_map(dtidx, latidx, lonidx, 4, w * sinF)

        if len(ais4s)!=0 and len(curs)==0:
            print(f'There is no curs. Target dtidx = {target_dtidx}, Num ais4 = {len(ais4)}')
        print(f'Num ais4: {len(ais4s)}, Num curs: {len(curs)}, Target dtidx: {target_dtidx}')
        Ais4ToCur.logout("Closing")
        Settei.close() # 偏流を計算

        Ais4ToCur.logout("Finished")
    
        svm.set_curs(curs)
        keys = ['Cur1', 'Cur2', 'X', 'Y', 'Lambda1', 'Lambda2', 'Phi1', 'Phi2']
        keys2 = ['cur1', 'cur2', 'n', 'e', 'lambda1', 'lambda2', 'phi1', 'phi2']
        saved_filenames = {}
        for i, key in enumerate(keys):
            dt_format = datetime(2011, 1, 1) + timedelta(hours=target_dtidx)
            saved_filenames[keys2[i]] = f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}{key}.csv"
        return saved_filenames

    @staticmethod
    def logout(message, linefeed=True):
        now = datetime.now()
        if not hasattr(Ais4ToCur, 'startTime') or Ais4ToCur.startTime == datetime.max:
            Ais4ToCur.startTime = now

        second_from_start = (now - Ais4ToCur.startTime).total_seconds()

        s = f"{now.strftime('%H:%M:%S')} , {int(second_from_start)} , {message}"
        #print(s + ("\r\n" if linefeed else "\r"), end='')
        #with open(Ais4ToCur.logFile, mode='a') as sw:
        #    sw.write(s + "\n")

    @staticmethod
    def parse_arguments():
        ap = argparse.ArgumentParser(description=f"Read {Ais4ToCur.inExt} file, calculate sea current values from AIS, and output {Ais4ToCur.outExt} file.")
        ap.add_argument("InFile", metavar="InFile", type=str, nargs="+", help=f"Input {Ais4ToCur.inExt} file(s)")
        ap.add_argument("-o", "--OutFolder", metavar="OutFolder", type=str, help=f"Folder to save output {Ais4ToCur.outExt} file(s)")
        return ap.parse_args()

class ais4:
    def __init__(self, MMSI, DtIdx, LatIdx, LonIdx, ThetaDeg, Flow, LineCount):
        self.MMSI = MMSI
        self.DtIdx = DtIdx
        self.LatIdx = LatIdx
        self.LonIdx = LonIdx
        self.ThetaDeg = ThetaDeg
        self.Flow = Flow
        self.LineCount = LineCount

class Cur:
    def __init__(self, MMSI, DtIdx, LatIdx, LonIdx, curN, curE):
        self.MMSI = MMSI
        self.DtIdx = DtIdx
        self.LatIdx = LatIdx
        self.LonIdx = LonIdx
        self.curN = curN
        self.curE = curE

class ShipVarManager:
    def __init__(self):
        self.init_var = 0.0
        self.out_folder = './'
        self.vars = {}
        self.errors = {}
        self.Curs = []
    
    def set_out_folder(self, out_folder):
        self.out_folder = out_folder
           
    def get_var(self, mmsi):
        if not mmsi in self.vars.keys():
            self.vars[mmsi] = self.init_var
        return self.vars[mmsi]
    
    def clear(self):
        self.vars = {}
        self.errors = {}
        self.Curs = []
    
    def set_curs(self, curs):
        self.Curs = curs
        
    def clear_curs(self):
        self.Curs = []
        
    def update(self, kalman_x, H, is_target, exist_data):
        for cur in self.Curs:
            grid0, grid1 = latlon_to_mesh(cur.LatIdx*deg_per_mesh, cur.LonIdx*deg_per_mesh)
            idx = int(kurosio_index[grid0][grid1]) # grid座標から黒潮のindex番号取得 黒潮領域外なら-1がくる
            try:
                if idx==-1:
                    #print(f'idx==-1 grid = {grid0}, {grid1}')
                    continue
                elif not is_target[idx]:
                    #print(f'not target idx:{idx}')
                    continue
            except:
                print(f'Error in ship var update. kurosio idx: {idx}')
            
            
            targets = np.where(is_target)[0] # フィルタリング対象のindex取得
            idxN = np.where(targets==idx)[0][0] # kalman_xにおけるindex取得
            idxE = idxN + len(targets)
            
            # v2のindex取得と、v2方向のphi2のcos, sin値の取得
            num_data = int((np.sum(exist_data)-1)/2)
            if not exist_data[idxN] or not exist_data[idxE]:
                print(f'Not exist data')
                continue
            if num_data!=(np.sum(exist_data)-1)/2:
                print(f'num_data is wrong value. {(np.sum(exist_data)-1)/2} should be int.')
            idxV2 = np.where(exist_data)[0]
            idxV2 = np.where(idxV2==idxN)[0][0]
            cos_phi2 = H[idxV2+num_data][idxV2]
            sin_phi2 = H[idxV2+num_data][idxV2+num_data]
            
            # v2方向の偏流の計算
            kalman_v2 = cos_phi2 * kalman_x[idxN][0] + sin_phi2 * kalman_x[idxE][0]
            cur_v2 = cos_phi2 * cur.curN + sin_phi2 * cur.curE

            if not cur.MMSI in self.errors:
                self.errors[cur.MMSI] = []
            
            error = kalman_v2 - cur_v2 # 誤差の計算
            self.errors[cur.MMSI].append(error)
            if error>10:
                print(f'error: {error}, kalman:{kalman_v2}, cur:{cur_v2}')
    
        
        for mmsi in self.errors.keys():
            if len(np_error) == 1:
                continue
            np_error = np.array(self.errors[mmsi])
            self.vars[mmsi] = np.sum(np_error*np_error)/(len(np_error)-1)
            
        self.clear_curs()
    
    def print_info(self):
        # 各船の分散を計算
        for mmsi in self.vars.keys():
            print(f'MMSI:{mmsi}, VAR:{self.vars[mmsi]}')
    
    def save_info(self, filename):
        # 各船の誤差と分散の情報を保存
        fname = osp.join(self.out_folder, 'error'+ filename + '.pkl')
        pkl.dump(self.errors, open(fname, 'wb'))
        print(f'Saved {fname}')
        fname = osp.join(self.out_folder, 'var' + filename + '.pkl')
        pkl.dump(self.vars, open(fname, 'wb'))
        print(f'Saved {fname}')
        

svm = ShipVarManager()

ais4_to_cur = Ais4ToCur()

class AISLoader:
# AISのファイルパスの読み取り
    def __init__(self, year, month, out_folder, pkl_path=None):
        self.year = year
        self.month = month
        self.base_dt = datetime(year, month, 1, 0, 0, 0)
        self.pathes = None
        self.keys = ['cur1', 'cur2', 'lambda1', 'lambda2', \
                     'phi1', 'phi2', 'n', 'e', 'd']
        self.out_folder = out_folder
        ais4_to_cur.set_outFolder(out_folder)
        if pkl_path:
            self.pkl_path = pkl_path
        else:
            self.pkl_path = r'E:/shunsukeE/data/ais/1509-ais4s-pkls'
        # self.cur_path = r'E:/shunsukeE/data/ais/ais_files'
        self.cur_path = self.pkl_path
        svm.out_folder = out_folder
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
    def load_path(self, keys=None):
        return None
    
    def set_keys(self, keys):
        self.keys = keys

    def get_ais4(self, dtidx):
        path = r'E:\shunsukeE\data\ais'
        dt_format = self.base_dt + timedelta(hours=dtidx)
        month = self.month
        day = dt_format.day
        hour = dt_format.hour + 1 #.ais4ファイルは1時から始まる(0時でない)
        #print(f'day: {day}, hour:{hour}, dtidx:{dtidx}')
        
        file_path = []
        if hour%2 == 0:
            if hour <= 14:
                i = int(hour/2) + 5
                filename = fr'japan_2015{month:02}{day:02}{i*2-2:02}0000-2015{month:02}{day:02}{i*2-1:02}5959.ais4'
                file_path.append(fr"{path}\1509{day:02}-{i}log\log\{filename}")
            else:
                i = int(hour/2) - 7
                
                if day+1==8 and (i==2 or i==3):
                    filename = fr'japan_2015{month:02}{day+1:02}{i*2-2:02}0000-2015{month:02}{day+1:02}{i*2-2:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i}-1log\log\{filename}")
                    filename = fr'japan_2015{month:02}{day+1:02}{i*2-1:02}0000-2015{month:02}{day+1:02}{i*2-1:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i}-2log\log\{filename}")
                else:
                    filename = fr'japan_2015{month:02}{day+1:02}{i*2-2:02}0000-2015{month:02}{day+1:02}{i*2-1:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i}log\log\{filename}")
        else:
            if hour < 15:
                i = int(hour/2) + 5
                filename = fr'japan_2015{month:02}{day:02}{i*2-2:02}0000-2015{month:02}{day:02}{i*2-1:02}5959.ais4'
                file_path.append(fr"{path}\15{month:02}{day:02}-{i}log\log\{filename}")
                filename = fr'japan_2015{month:02}{day:02}{(i+1)*2-2:02}0000-2015{month:02}{day:02}{(i+1)*2-1:02}5959.ais4'
                file_path.append(fr"{path}\15{month:02}{day:02}-{i+1}log\log\{filename}")
            elif hour > 15:
                i = int(hour/2) - 7

                
                if day+1==8 and (i==2 or i==3):
                    filename = fr'japan_2015{month:02}{day+1:02}{i*2-2:02}0000-2015{month:02}{day+1:02}{i*2-2:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i}-1log\log\{filename}")
                    filename = fr'japan_2015{month:02}{day+1:02}{i*2-1:02}0000-2015{month:02}{day+1:02}{i*2-1:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i}-2log\log\{filename}")
                else:
                    filename = fr'japan_2015{month:02}{day+1:02}{i*2-2:02}0000-2015{month:02}{day+1:02}{i*2-1:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i}log\log\{filename}")

                
                if day+1==8 and (i+1==2 or i+1==3):
                    filename = fr'japan_2015{month:02}{day+1:02}{(i+1)*2-2:02}0000-2015{month:02}{day+1:02}{(i+1)*2-2:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i+1}-1log\log\{filename}")
                    filename = fr'japan_2015{month:02}{day+1:02}{(i+1)*2-1:02}0000-2015{month:02}{day+1:02}{(i+1)*2-1:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i+1}-2log\log\{filename}")
                else:
                    filename = fr'japan_2015{month:02}{day+1:02}{(i+1)*2-2:02}0000-2015{month:02}{day+1:02}{(i+1)*2-1:02}5959.ais4'
                    file_path.append(fr"{path}\15{month:02}{day+1:02}-{i+1}log\log\{filename}")                          
            else:
                filename = fr'japan_2015{month:02}{day:02}{12*2-2:02}0000-2015{month:02}{day:02}{12*2-1:02}5959.ais4'
                file_path.append(fr"{path}\15{month:02}{day:02}-{12}log\log\{filename}")
                filename = fr'japan_2015{month:02}{day+1:02}{1*2-2:02}0000-2015{month:02}{day+1:02}{1*2-1:02}5959.ais4'
                file_path.append(fr"{path}\15{month:02}{day+1:02}-{1}log\log\{filename}")
        #print(f'file_path: {file_path}')
        return file_path
    
    def load_ais_dtidx(self, dtidx, keys=None, use_pkl=True):
        if not keys:
            keys = self.keys
            
        def load_ais_data(ais_path):
            try:
                ais_data = pd.read_csv(ais_path, encoding="cp932", header=None)
                ais_data = ais_data.values
            except:
                print(f'Error load {ais_path}')
            return ais_data

         
        dt_format = self.base_dt + timedelta(hours=dtidx)
        dtidx2 = date_to_dtidx(datetime(2011, 1, 1), dt_format)
        
        if not use_pkl:
            file_path = self.get_ais4(dtidx)   
            ais4_to_cur.load_ais4(file_path, dtidx2)
        else:
            ais4_to_cur.load_ais4s(self.pkl_path, dt_format)
        saved_filenames = ais4_to_cur.calc_cur(dtidx2) 
        
        data = {}
        for key in keys:
            data[key] = load_ais_data(saved_filenames[key]) 
        return data
    
    def load_ais_day(self, day, keys=None, use_pkl=True):
        if not keys:
            keys = self.keys
            
        data = {}
        for key in keys:
            data[key] = {}
            
        for hour in range(24):
            dt = datetime(self.year, self.month, day, hour, 0, 0)
            dtidx = date_to_dtidx(self.base_dt, dt)     
            dt_format = self.base_dt + timedelta(hours=dtidx)  
            dtidx2 = date_to_dtidx(datetime(2011, 1, 1), dt_format)
            
            if not use_pkl:
                file_path = self.get_ais4(dtidx)
                ais4_to_cur.load_ais4(file_path, dtidx2)
            else:
                ais4_to_cur.load_ais4s(self.pkl_path, dt_format)
            saved_filenames = ais4_to_cur.calc_cur(dtidx2)
            
            for key in keys:
                data[key][dtidx] = pd.read_csv(saved_filenames[key], encoding="cp932", header=None)
                data[key][dtidx] = data[key][dtidx].values
                # try:
                #     data[key][dtidx] = pd.read_csv(saved_filenames[key], encoding="cp932", header=None)
                #     data[key][dtidx] = data[dtidx].values
                # except:
                #     print(f'Error load {saved_filenames[key]}')
        return data

    def load_cur(self, dtidx, keys=None):
        key_map = {'cur1': 'Cur1', 
                   'cur2': 'Cur2', 
                   'lambda1': 'Lambda1', 
                   'lambda2': 'Lambda2',
                   'phi1': 'Phi1', 
                   'phi2': 'Phi2',
                   'n': 'X',
                   'e': 'Y',
                   'd': 'D',
                   }

        if not keys:
            keys = self.keys

        data = {}
        for key in keys:
            data[key] = [] 

        dt_format = dtidx_to_date(self.base_dt, dtidx)
        day = dt_format.day
        hour = dt_format.hour
        for key in keys:
            pathName = rf'AisCurr{self.year}{self.month:02}{day:02}{hour:02}{key_map[key]}.csv'    
            pathName = osp.join(self.cur_path, pathName)
            data[key] = pd.read_csv(pathName, encoding="cp932", header=None)
            data[key] = data[key].values
            print(f'{key}: shape{data[key].shape}')

        return data 
    
    def load_test(self, day):
        for hour in range(24):
            dt = datetime(self.year, self.month, day, hour, 0, 0)
            dtidx = date_to_dtidx(self.base_dt, dt)   
            dt_format = self.base_dt + timedelta(hours=dtidx)  
            dtidx2 = date_to_dtidx(datetime(2011, 1, 1), dt_format)
              
            file_path = self.get_ais4(dtidx)
            ais4_to_cur.load_ais4(file_path, dtidx2)
    
    def create_ais4s_file(self, day):      
            
        for hour in range(24):
            dt = datetime(self.year, self.month, day, hour, 0, 0)
            dtidx = date_to_dtidx(self.base_dt, dt)     
            dt_format = self.base_dt + timedelta(hours=dtidx)  
            dtidx2 = date_to_dtidx(datetime(2011, 1, 1), dt_format)
              
            file_path = self.get_ais4(dtidx)
            ais4_to_cur.load_ais4(file_path, dtidx2)

            ais4_to_cur.dump_ais4(dt, self.pkl_path)

if __name__ == "__main__":
    #pm.printline('checking Ais4ToCur class')
    #args = Ais4ToCur.parse_arguments()
    #inFiles = args.InFile
    #outFolder = args.OutFolder
    
    # Check KurosioRangePluse class
    if False: 
        HalfLat = 0.1
        HalfLon = 0.2
        ln2 =  0.6931471805599453094172321
        Thres = 0.05
        degPerMesh = deg_per_mesh
        hrsPerMesh = 1.0
        krp = KurosioRangePluse(-HalfLon * math.log(Thres) / ln2, -HalfLat * math.log(Thres) / ln2)
        krp.print_info()
        tests = [[30, 136], [20, 120], [50, 130], [32, 120], [33, 140]]
        for test in tests:
            print(f'{test}: {krp.test(test[0], test[1])}')
            
    # Check Ais4ToCur class
    if False:
        inFiles = [r'E:\shunsukeE\data\ais\150901-6log\log\japan_20150901100000-20150901115959.ais4'] # ais4ファイルのパス
        outFolder = r'data/ais_test' # 出力先パス
        atc = Ais4ToCur()
        atc.load_ais4(inFiles, outFolder) # ais4ファイルの読み込み
        
        # 偏流の計算
        dtidx = atc.ais4s[0].DtIdx #ターゲットの時刻
        saved_filenames = atc.calc_cur(dtidx) # 固有値方向の偏流, 固有値, 固有ベクトルの角度, 北方向の偏流, 東方向の偏流を計算しcsvファイルを出力し、保存先のパスを返す
        svm.print_info()

    # Check AISLoader class
    if False:
        #pm.printline('checking AISLoader class')
        use_pkl = True
        year = 2015
        month = 9
        out_folder = r'data/ais_test'
        ais_keys = ['cur1', 'cur2', 'lambda1', 'lambda2', 'phi1', 'phi2', 'n', 'e']
        al = AISLoader(year, month, out_folder)
        al.set_keys(ais_keys)
        
        #pm.printline('testing load_ais_dtidx func in AISLoader class')
        #data = al.load_ais_dtidx(10)
        #pm.printline('testing load_ais_day func in AISLoader class')
        #data = al.load_ais_day(10)
        for i in range(1, 2):
            print(f'day: {i}')
            al.load_ais_day(i, use_pkl=False)
            print('\n')

    # Create ais4s pkl files
    if True:
        # pm.printline('checking AISLoader class')
        # 引数で年、月、出力先フォルダ、pklの読み込み先を設定
        year = 2015
        month = 9
        #out_folder = path_ais 
        out_folder = 'test' 
        al = AISLoader(year, month, out_folder, pkl_path=out_folder)
        
        # cur1: 固有値方向の偏流１, cur2：固有値方向の偏流２, lambda1:固有値１, lambda2:固有値２, phi1:固有ベクトルの角度１, phi2:固有ベクトルの角度２, n:北方向の偏流, e:東方向の偏流
        ais_keys = ['cur1', 'cur2', 'lambda1', 'lambda2', 'phi1', 'phi2', 'n', 'e']
        al.set_keys(ais_keys) # 使用するkeyの設定
        
        # def check_diff_nanmap():
        #     ais_nan_map_path = osp.join(out_folder, 'AisNanmap.csv')
        #     ais_nan_map = pd.read_csv(ais_nan_map_path, encoding="cp932", header=None)
        #     ais_nan_map = ais_nan_map.values
        #     if ais_nan_map.shape[0] != nan_map_pooled.shape[0]\
        #         or ais_nan_map.shape[1] != nan_map_pooled.shape[1]:
        #             print(f'Not match shape AIS:{ais_nan_map.shape}, NanMap{nan_map_pooled.shape}')
        #     diffmap = np.zeros(ais_nan_map.shape)
        #     tf1 = ais_nan_map==ais_nan_map
        #     tf2 = nan_map_pooled==nan_map_pooled
        #     for i in range(ais_nan_map.shape[0]):
        #         for j in range(ais_nan_map.shape[1]):
        #             if tf1[i][j] and tf2[i][j]:
        #                 diffmap[i][j] = 2
        #             elif tf1[i][j] and not tf2[i][j]:
        #                 diffmap[i][j] = 1 
        #             elif not tf1[i][j] and tf2[i][j]:
        #                 diffmap[i][j] = -1 
        #             elif not tf1[i][j] and not tf2[i][j]:
        #                 diffmap[i][j] = 0.0 
        #             else:
        #                 print(tf1[i][j])
        #                 print(tf2[i][j])
        #     df = pd.DataFrame(diffmap)
        #     path = osp.join(out_folder, 'diff_nanmap.csv')
        #     df.to_csv(path, index=False, header=False)
                        
                    
            
        n_day = nday_month(month) 
        Settei.init(0, 0, out_folder)
        ais4_to_cur.save_nanmap()
        #check_diff_nanmap()

        for i in range(1, n_day+1):
            print(f'day: {i}')
            #al.load_test(i)
            al.create_ais4s_file(i)
            al.load_ais_day(i)
            
