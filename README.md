# Table of Content
- [Table of Content](#table-of-content)
- [Introduction](#introduction)
- [How It Works](#how-it-works)
  - [事前準備](#事前準備)
  - [プログラムの実施例](#プログラムの実施例)


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
A1～A5までのProgram.exeを実行する．
実行例
```
cd ~/nmri/NMEA0183_decorder/AIS_Decorder
A1-AIS_ToyoJAXAFileoutToAis1/Program.exe $(ls ../../data/ais/*/log/*.log | sort -V)
```

**評価用データの用意**
評価用データとして，船のログから計算した偏流値を使用する．偏流値の作成手順は以下のようにする．
1. 船のログ内の必要なNMEAデータを抜き出したファイルslog1を作成する.
```
cd ~/nmri/NMEA0183_decorder
S1-ShipLogToS1/Program.exe ...
```
2. slog1ファイルから偏流値を計算し，csvファイルに書き出す
```
cd ~/nmri/KalmanFilter
python -m CreateShipCurProgram
```
(csvファイルの内容はtidx, UTC, CurN, CurE, Grid0, Grid1, Lat, Lonの順で各行出力される.)

## プログラムの実施例
**パラメータ設定**
パラメータは`config/config.ini`内の
`[KALMAN_PARAM]`でカルマンフィルタのパラメータ，
`[ANALYSIS_PARAM]`で分析用のパラメータを設定する．

**カルマンフィルタの実行**
```
cd ~/nmri/KalmanFilter
python -m KalmanFilterProgram
```

**カルマンフィルタのログの分析**
```
cd ~/nmri/KalmanFilter
python -m AnalysisProgram 
```




<!--# Usage -->


