import numpy as np
import pandas as pdo
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 相関行列
def corr(data, save=True, save_name="corr"):
    # 相関行列の計算
    correlation_matrix = data.corr()

    # 結果の表示
    print("相関行列:")
    print(correlation_matrix)

    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1)
    plt.title("相関行列")
    plt.savefig(f"./logs/{save_name}.png")
    return correlation_matrix

# 平均絶対誤差
def mae(target, data):
    res = []
    for key in data.keys():
        res.append(mean_absolute_error(target, data[key]))
        print("MAE:")
        print(res[-1])
    return res 
    
# 平均二乗誤差
def mse(target, data):
    res = []
    for key in data.keys():
        res.append(mean_squared_error(target, data[key]))
        print("MAE:")
        print(res[-1])
    return res 
