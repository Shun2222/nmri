import argparse
import os
import math
from datetime import datetime, timedelta
import csv

class Settei:
    LatIdxMax = 1800
    LatIdxMin = 750
    LonIdxMax = 5401
    LonIdxMin = 4211
    DtIdxMax = 0
    DtIdxMin = 0
    tmpDtIdxMin = 0
    OutFolder = ""
    HalfLat = 0.1
    HalfLon = 0.2
    HalfHrs = 0.1
    ln2 =  0.6931471805599453094172321
    Thres = 0.05
    degPerMesh = 1.0 / 36
    hrsPerMesh = 1.0
    LatRange = int(-HalfLat * math.log(Thres) / ln2 / degPerMesh)
    LonRange = int(-HalfLon * math.log(Thres) / ln2 / degPerMesh)
    HrsRange = int(-HalfHrs * math.log(Thres) / ln2 / hrsPerMesh)
    h = HrsRange * 2 + 1
    weights = None
    Map = None
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

        Settei.LatLonIsValid = [
            [False for _ in range(Settei.LonIdxMin, Settei.LonIdxMax + 1)] for _ in range(Settei.LatIdxMin, Settei.LatIdxMax + 1)]

        with open("data/cur.csv", "r") as sr:
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
        while Settei.tmpDtIdxMin + Settei.h <= DtIdx:
            Settei.map_to_csv_and_clear(Settei.tmpDtIdxMin)
            Settei.tmpDtIdxMin += 1

        Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][item] += Value

    @staticmethod
    def close():
        for DtIdx in range(Settei.tmpDtIdxMin, Settei.DtIdxMax + 1):
            Settei.map_to_csv_and_clear(DtIdx)

    @staticmethod
    def is_lat_lon_valid(LatIdx, LonIdx):
        return Settei.LatLonIsValid[LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin]

    @staticmethod
    def map_to_csv_and_clear(DtIdx):
        dt_format = datetime(2011, 1, 1) + timedelta(hours=DtIdx)
        print(f"Output : {dt_format.strftime('%Y%m%d%H')}")
        with open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Cur1.csv", "w") as sw_cur1, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Cur2.csv", "w") as sw_cur2, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}X.csv", "w") as sw_x, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Y.csv", "w") as sw_y, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Lambda1.csv", "w") as sw_lambda1, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Lambda2.csv", "w") as sw_lambda2, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Phi1.csv", "w") as sw_phi1, \
                open(f"{Settei.OutFolder}/AisCurr{dt_format.strftime('%Y%m%d%H')}Phi2.csv", "w") as sw_phi2:

            for LatIdx in range(Settei.LatIdxMax, Settei.LatIdxMin - 1, -1):
                s_cur1 = ""
                s_cur2 = ""
                s_x = ""
                s_y = ""
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
                        X = B1/Lambda1
                        Y = B2/Lambda2
                        Cur1 = (X * math.cos(Phi1) + Y * math.sin(Phi1))
                        Cur2 = (X * math.cos(Phi2) + Y * math.sin(Phi2))
                        #s_cur1 += f"{Cur1},"
                        #s_cur2 += f"{Cur2},"
                        #s_x += f"{X},"
                        #s_y += f"{Y},"
                        #s_lambda1 += f"{Lambda1},"
                        #s_lambda2 += f"{Lambda2},"
                        #s_phi1 += f"{Phi1},"
                        #s_phi2 += f"{Phi2},"
                        s_cur1 += f"{Cur1:.2f},"
                        s_cur2 += f"{Cur2:.2f},"
                        s_x += f"{X:.2f},"
                        s_y += f"{Y:.2f},"
                        s_lambda1 += f"{Lambda1:.2f},"
                        s_lambda2 += f"{Lambda2:.2f},"
                        s_phi1 += f"{Phi1:.2f},"
                        s_phi2 += f"{Phi2:.2f},"
                    else:
                        s_cur1 += f","
                        s_cur2 += f","
                        s_x += f","
                        s_y += f","
                        s_lambda1 += f","
                        s_lambda2 += f","
                        s_phi1 += f","
                        s_phi2 += f","

                sw_cur1.write(s_cur1[:-1] + "\n")
                sw_cur2.write(s_cur2[:-1] + "\n")
                sw_x.write(s_x[:-1] + "\n")
                sw_y.write(s_y[:-1] + "\n")
                sw_lambda1.write(s_lambda1[:-1] + "\n")
                sw_lambda2.write(s_lambda2[:-1] + "\n")
                sw_phi1.write(s_phi1[:-1] + "\n")
                sw_phi2.write(s_phi2[:-1] + "\n")
        for LatIdx in range(Settei.LatIdxMax, Settei.LatIdxMin - 1, -1):
            for LonIdx in range(Settei.LonIdxMin, Settei.LonIdxMax + 1):
                for i in range(5):
                    Settei.Map[DtIdx % Settei.h][LatIdx - Settei.LatIdxMin][LonIdx - Settei.LonIdxMin][i] = 0
class Program:
    inExt = "ais4"
    outExt = "csv"
    logFile = "logout.log"
    startTime = datetime.max

    @staticmethod
    def main():
        args = Program.parse_arguments()
        inFiles = args.InFile
        outFolder = args.OutFolder
        if not outFolder:
            outFolder = "data/ais"

        dtMin = float('inf')
        dtMax = float('-inf')
        ais4s = []
        Program.logout("Start Reading")

        for inFile in inFiles:
            with open(inFile, "r") as sr:
                sr.readline()  # Skip first line
                sr.readline()  # Skip header
                l = 0
                while True:
                    line = sr.readline()
                    if not line:
                        break
                    print("{:.2f}%,{}lines done\r".format(100.0 * sr.tell() / os.path.getsize(inFile), l), end="")
                    l += 1
                    ss = line.split(',')

                    MMSI = int(ss[0])
                    DtIdx = int(ss[1])
                    LatIdx = int(ss[2])
                    LonIdx = int(ss[3])
                    ThetaDeg = float(ss[4])
                    F = float(ss[5])
                    LineCount = float(ss[6])

                    dtMin = min(dtMin, DtIdx)
                    dtMax = max(dtMax, DtIdx)

                    ais4s.append(ais4(MMSI, DtIdx, LatIdx, LonIdx, ThetaDeg, F, LineCount))

        Program.logout("Sort")
        ais4s.sort(key=lambda x: (x.DtIdx, x.LatIdx, x.LonIdx, x.MMSI))

        Program.logout("Start Map")

        Settei.init(dtMax, dtMin, outFolder)

        for i, a in enumerate(ais4s):
            print("{:.2f}%,{}lines done\r".format(100.0 * i / len(ais4s), i), end="")

            theta = a.ThetaDeg * (math.pi / 180)
            sin = math.sin(theta)
            cos = math.cos(theta)
            coscos = cos * cos
            sincos = sin * cos
            sinsin = sin * sin
            cosF = cos * a.Flow
            sinF = sin * a.Flow

            for dtidx in range(a.DtIdx - Settei.HrsRange, a.DtIdx + Settei.HrsRange + 1):
                for latidx in range(a.LatIdx - Settei.LatRange, a.LatIdx + Settei.LatRange + 1):
                    for lonidx in range(a.LonIdx - Settei.LonRange, a.LonIdx + Settei.LonRange + 1):
                        if Settei.LatIdxMin <= latidx <= Settei.LatIdxMax and \
                                Settei.LonIdxMin <= lonidx <= Settei.LonIdxMax and \
                                Settei.DtIdxMin <= dtidx <= Settei.DtIdxMax and \
                                Settei.is_lat_lon_valid(latidx, lonidx):
                            w = Settei.weight(latidx - a.LatIdx, lonidx - a.LonIdx, dtidx - a.DtIdx) * a.LineCount
                            Settei.add_map(dtidx, latidx, lonidx, 0, w * coscos)
                            Settei.add_map(dtidx, latidx, lonidx, 1, w * sincos)
                            Settei.add_map(dtidx, latidx, lonidx, 2, w * sinsin)
                            Settei.add_map(dtidx, latidx, lonidx, 3, w * cosF)
                            Settei.add_map(dtidx, latidx, lonidx, 4, w * sinF)

        Program.logout("Closing")
        Settei.close()

        Program.logout("Finished")

    @staticmethod
    def logout(message, linefeed=True):
        now = datetime.now()
        if not hasattr(Program, 'startTime') or Program.startTime == datetime.max:
            Program.startTime = now

        second_from_start = (now - Program.startTime).total_seconds()

        s = f"{now.strftime('%H:%M:%S')} , {int(second_from_start)} , {message}"
        print(s + ("\r\n" if linefeed else "\r"), end='')
        with open(Program.logFile, mode='a') as sw:
            sw.write(s + "\n")

    @staticmethod
    def parse_arguments():
        ap = argparse.ArgumentParser(description=f"Read {Program.inExt} file, calculate sea current values from AIS, and output {Program.outExt} file.")
        ap.add_argument("InFile", metavar="InFile", type=str, nargs="+", help=f"Input {Program.inExt} file(s)")
        ap.add_argument("-o", "--OutFolder", metavar="OutFolder", type=str, help=f"Folder to save output {Program.outExt} file(s)")
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

if __name__ == "__main__":
    Program.main()
