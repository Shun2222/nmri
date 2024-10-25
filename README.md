# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [How It Works](#how-it-works)
  - [事前準備](#事前準備)
  - [プログラムの実施例](#プログラムの実施例)
- [Folder Structure](#folder-structure)


# Introduction
衛星データによる偏流予測値をAISデータを使用してカルマンフィルタにより修正する．
プログラムを動かすまでの手順は[How It Works](#-howitworks)内で説明され，
計算式の詳細は`KalmanFilter/README.md`で記述される．


# How It Works 
## 事前準備
**必要なプログラムの用意**
```
mkdir ~/workspace
cd ~/workspace
git clone https://github.com/shun2222/nmri.git
git submodule update -i
```

**衛星データの用意**
JCOPEデータを範囲xxx~xxxで1マス1/36度で偏流値のcsvファイルを用意する．

**AISデータの用意**
NMEA0183_decorder/READEME.md内How It Worksを実施する．

**評価用データの用意**
評価用データとして，船のログから計算した偏流値を使用する．偏流値の作成手順は以下のようにする．
1. 船のログ内の必要なNMEAデータを抜き出したファイルslog1を作成する.
```
cd ~/workspace/nmri/NMEA0183_decorder
S1-ShipLogToS1/Program.exe {ShiplogDir}/*.log
```
2. slog1ファイルから偏流値を計算し，csvファイルに書き出す
```
cd ~/workspace/nmri/KalmanFilter
python -m CreateShipCurProgram
```
(csvファイルの内容はtidx, UTC, CurN, CurE, Grid0, Grid1, Lat, Lonの順で各行出力される.)

## プログラムの実施例
**環境構築 (Anaconda推奨)**
- Docker
```
cd ~/workspace/nmri/KalmanFilter/DockerFile
docker build --target base -t kalmanfilter -f Dockerfile.base .
docker run -it --rm --name kalmanfilter_container -v /mnt/shunsukeUeki:/mnt/shunsukeUeki kalmanfilter /bin/bash
docker exec -it kalmanfilter_container /bin/bash
```

- Anaconda
```
cd ~/workspace/nmri/KalmanFilter/AnacondaFile
conda kalmanfilter create -f kalmanfilter.yaml
conda activate kalmanfilter 
```

**パラメータ設定**
パラメータは`config/config.ini`内の
`[KALMAN_PARAM]`でカルマンフィルタのパラメータ，
`[ANALYSIS_PARAM]`で分析用のパラメータを設定する．

**カルマンフィルタの実行**
```
cd ~/workspace/nmri/KalmanFilter
python -m KalmanFilterProgram
```

**カルマンフィルタのログの分析**
```
cd ~/workspace/nmri/KalmanFilter
python -m AnalysisProgram 
```

# Folder Structure
`tree -d -L 3`で表示

```
|-- KalmanFilter
|   |-- AnacondaFile
|   |-- AnalysisProgram
|   |-- CreateAisCurProgram
|   |-- CreoateShipCurProgram
|   |-- DockerFile
|   |-- KalmanFilterProgram
|   |   `-- utils
|   |-- config
|   |-- logs
|   |-- programの墓場
|   |   |-- analysis_programs
|   |   |-- check_programs
|   |   |-- old_programs
|   |   `-- shiplog_generator
|   |-- test
|   `-- utils
|-- NMEA0183_decorder
|   |-- AIS_Decorder
|   |   |-- A1-AIS_ToyoJAXAFileoutToAis1
|   |   |-- A2-AIS1ToAis2ManyFileRead
|   |   |-- A3-Ais2ToAis3
|   |   |-- A3-Ais2ToAis3-remove
|   |   |-- A4-Ais3-removeToMap
|   |   |-- A4-Ais3ToAis4
|   |   |-- A5-Ais4ToAisCurrLowMemConsumption
|   |   |-- A5_2-Ais4_2ToAisCurrLowMemConsumption
|   |   |-- A6-Ais4ToParameterSpeedUpCost
|   |   |-- K1-Ais4ToCurVecForKalman
|   |   |-- S-Ais4ToEachShipCurVecForKalman
|   |   `-- oldfiles
|   |-- JCOPE_decorder
|   |-- S1-ShipLogToS1
|   |   |-- Properties
|   |   `-- obj
|   `-- images
```




<!--# Usage -->


