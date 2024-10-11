# Introduction
衛星データによる偏流予測値をAISデータを使用してカルマンフィルタにより修正する．

# How It Works 
## 事前準備
**衛星データの用意**
JCOPEデータを範囲xxx~xxxで1マス1/36度で偏流値のcsvファイルを用意する．

**AISデータの用意**
xxx

## プログラムの実施例
**パラメータ設定**
パラメータは`config/config.ini`内の
`[KALMAN_PARAM]`でカルマンフィルタのパラメータ，
`[ANALYSIS_PARAM]`で分析用のパラメータを設定する．

**カルマンフィルタの実行**
```
cd KalmanFilter
python -m KalmanFilterProgram
```

**カルマンフィルタのログの分析**
```
cd KalmanFilter
python -m AnalysisProgram 
```




<!--# Usage -->


