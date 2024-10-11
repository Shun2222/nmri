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
target_ships  = {'中春': '中春丸',
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
                '立眞': '立眞丸'}


year = dt_year = 2015
month = dt_month = 9
dt_day = 1
n_day = nday_month(dt_month)
done_ship = []

for target_ship in target_ships.keys():
    print(f'{target_ship}')
    if target_ship in done_ship:
        print(f'skip {target_ship}')
        continue

    path2 = osp.join(path, target_ship, '2015')
    for day in range(1, n_day+1):
        f = fr'{target_ships[target_ship]}FileOut{year}{month:02}{day:02}.slog1'
        isfile = os.path.isfile(osp.join(path2, f))
        if not isfile:
            print(f'Not Exist {f}')
        else:
            try:
                log = pd.read_csv(osp.join(path2, f), encoding="cp932", header=None)
                log.columns = ["UTC", "NMEA", "data1", "data2"]
                print(len(log))
            except:
                print(f'Error load {f}')

