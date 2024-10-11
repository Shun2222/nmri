import tqdm
import json
import datetime
import configparser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pkl
import ast
import re
import logger 

config_ini = configparser.ConfigParser()
config_ini.optionxform = str
config_ini.read('../config.ini', encoding='utf-8')

# コメントを取り除くための正規表現
def remove_comment(value):
    # '#' または ';' 以降の部分を削除
    value = re.sub(r'\s*#.*', '', value)
    value = re.sub(r'\s*;.*', '', value)
    return value.strip()

def config_get(in1, in2):
    return remove_comment(config_ini.get(in1, in2))

def config_get_with_log(in1, in2):
    s = remove_comment(config_ini.get(in1, in2))
    logger.record_tabular(in2, s)
    return s

    
MODE = json.loads(config_get_with_log("PARAM", "MODE"))
Q_values = ast.literal_eval(config_get_with_log("PARAM", "Q_VALUES"))
N_lambda = int(config_get_with_log("PARAM", "N_LAMBDA"))
Min_lambda = int(config_get_with_log("PARAM", "MIN_LAMBDA"))
PATH_SAVE = path_save = json.loads(config_get_with_log("PARAM", "PATH_SAVE"))

pool_size = int(config_get_with_log(MODE, "POOL_SIZE"))
kurosio_lat_range1 = ast.literal_eval(config_get_with_log(MODE, "KUROSIO_LAT_RANGE1"))
kurosio_lat_range2 = ast.literal_eval(config_get_with_log(MODE, "KUROSIO_LAT_RANGE2"))
kurosio_lon_range = ast.literal_eval(config_get_with_log(MODE, "KUROSIO_LON_RANGE"))
PATH_AIS = path_ais = json.loads(config_get_with_log(MODE, "PATH_AIS"))
PATH_JCOPE = path_jcope = json.loads(config_get_with_log(MODE, "PATH_JCOPE"))

print(type(kurosio_lat_range1[0]))
print(kurosio_lat_range1[1])
print(kurosio_lat_range2[0])
print(kurosio_lat_range2[1])
print(type(kurosio_lon_range[0]))
print(kurosio_lon_range[1])

logger.dump_tabular()

ais_lat_max = 1800
ais_lat_min = 750
ais_lon_max = 5401
ais_lon_min = 4211
lat_range = [20-1/36, 50+1/36]
lon_range = [117-1/36, 150+1/36]
map_size_ais = [1050, 1191]
map_size_jcope = [1082, 1191]

deg_per_mesh = 1/(36/pool_size)
map_pooled_size = (int(map_size_ais[0]/pool_size), int(map_size_ais[1]/pool_size))

line1 = [[kurosio_lon_range[0], kurosio_lat_range1[0]], [kurosio_lon_range[1], kurosio_lat_range2[0]]]
line2 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]
line4 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[0], kurosio_lat_range2[0]]]
line3 = [[kurosio_lon_range[1], kurosio_lat_range2[0]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]

kurosio_latidx_range1 = [int((kurosio_lat_range1[0]-lat_range[0])/deg_per_mesh), int((kurosio_lat_range1[1]-lat_range[0])/deg_per_mesh)]
kurosio_latidx_range2 = [int((kurosio_lat_range2[0]-lat_range[0])/deg_per_mesh), int((kurosio_lat_range2[1]-lat_range[0])/deg_per_mesh)]
kurosio_lonidx_range = [int((kurosio_lon_range[0]-lon_range[0])/deg_per_mesh), int((kurosio_lon_range[1]-lon_range[0])/deg_per_mesh)]