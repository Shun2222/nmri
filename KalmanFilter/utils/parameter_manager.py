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
import utils.logger as logger
import os.path as osp

config_ini = configparser.ConfigParser()
config_ini.optionxform = str
config_path = r'C:\Users\nmri\Documents\shunsuke\nmri\KalmanFilter/config'
config_ini.read(osp.join(config_path, 'config.ini'), encoding='utf-8')
params = {}

def remove_comment(value):
    # '#' または ';' 以降の部分を削除
    value = re.sub(r'\s*#.*', '', value)
    value = re.sub(r'\s*;.*', '', value)
    return value.strip()

def config_get(in1, in2):
    return remove_comment(config_ini.get(in1, in2))


def load_json_line_value(file_path, key):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                json_object = json.loads(line)
                if key in json_object: 
                    data.append(json_object[key])
    except:
        assert False, f"Cannot open {file_path} or cannot find {key} in the path"
    return data

def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data


class ParameterManager():
    def __init__(self, param_name=''):
        self.param_name = param_name
        self.params = {}

    def config_get_set_param(self, in1, in2):
        s = remove_comment(config_ini.get(in1, in2))
        self.params[in2] = s
        #logger.record_tabular(in2, s)
        return s

    def config_getboolean_set_param(self, in1, in2):
        #s = config_ini.getboolean(in1, in2)
        s = remove_comment(config_ini.get(in1, in2))
        if 'true' in s:
            s = True
        else:
            s = False
        self.params[in2] = s
        #logger.record_tabular(in2, s)
        return s

    def dump_params(self):
        for key in params:
            logger.record_tabular(key, self.params[key])
        logger.dump_tabular()

    def get_param(self):
        if self.param_name=='kalman':
            self.get_kalman_param()
        elif self.param_name=='analysis':
            self.get_analysis_param()
        elif self.param_name=='shipLog':
            self.get_shipLog_param()
        #else:
            #self.get_area_param()
        return self.params

    def get_kalman_param(self):
        Q_values = ast.literal_eval(self.config_get_set_param("KALMAN_PARAM", "Q_VALUES"))
        N_lambda = int(self.config_get_set_param("KALMAN_PARAM", "N_LAMBDA"))
        N_D = int(self.config_get_set_param("KALMAN_PARAM", "N_D"))
        Min_lambda1 = int(self.config_get_set_param("KALMAN_PARAM", "MIN_LAMBDA1"))
        Min_lambda2 = int(self.config_get_set_param("KALMAN_PARAM", "MIN_LAMBDA2"))
        Min_D = int(self.config_get_set_param("KALMAN_PARAM", "MIN_D"))
        path_save = json.loads(self.config_get_set_param("KALMAN_PARAM", "PATH_SAVE"))
        USE_SHIPVAR = self.config_getboolean_set_param("KALMAN_PARAM", "USE_SHIPVAR")
        USE_AIS_MEDIAN = self.config_getboolean_set_param("KALMAN_PARAM", "USE_AIS_MEDIAN")
        USE_AIS_REMOVE_BAD_MMSI = self.config_getboolean_set_param("KALMAN_PARAM", "USE_AIS_REMOVE_BAD_MMSI")

        AREA = json.loads(self.config_get_set_param("KALMAN_PARAM", "AREA"))
        self.get_area_param(AREA)

    def get_analysis_param(self):
        path_log = json.loads(self.config_get_set_param("ANALYSIS_PARAM", "PATH_LOG"))
        path_ship = json.loads(self.config_get_set_param("ANALYSIS_PARAM", "PATH_SHIP"))
        DAY = int(json.loads(self.config_get_set_param("ANALYSIS_PARAM", "DAY")))
        HOUR = int(json.loads(self.config_get_set_param("ANALYSIS_PARAM", "HOUR")))
        MAX_DAY = int(json.loads(self.config_get_set_param("ANALYSIS_PARAM", "MAX_DAY")))
        MAX_HOUR = int(json.loads(self.config_get_set_param("ANALYSIS_PARAM", "MAX_HOUR")))
        USE_PICKLE = self.config_getboolean_set_param("ANALYSIS_PARAM", "USE_PICKLE")
        AIS_JCOPE_TEST = self.config_getboolean_set_param("ANALYSIS_PARAM", "AIS_JCOPE_TEST")
        VISUALIZE = self.config_getboolean_set_param("ANALYSIS_PARAM", "VISUALIZE")
        EVALUATION = self.config_getboolean_set_param("ANALYSIS_PARAM", "EVALUATION")

        AREA = load_json_line_value(osp.join(path_log, 'progress.json'), "AREA")[0]
        if AREA[0]=='"':
            AREA = AREA[1:-1] 
        self.params['AREA'] = AREA

        USE_AIS_REMOVED_BAD_MMSI = load_json_line_value(osp.join(path_log, 'progress.json'), "USE_AIS_REMOVE_BAD_MMSI")[0]
        self.params['USE_AIS_REMOVED_BAD_MMSI'] = USE_AIS_REMOVED_BAD_MMSI
        self.get_area_param(AREA)

    def get_area_param(self, AREA):
        pool_size = int(self.config_get_set_param(AREA, "POOL_SIZE"))
        kurosio_lat_range1 = ast.literal_eval(self.config_get_set_param(AREA, "KUROSIO_LAT_RANGE1"))
        kurosio_lat_range2 = ast.literal_eval(self.config_get_set_param(AREA, "KUROSIO_LAT_RANGE2"))
        kurosio_lon_range = ast.literal_eval(self.config_get_set_param(AREA, "KUROSIO_LON_RANGE"))
        PATH_AIS = path_ais = json.loads(self.config_get_set_param(AREA, "PATH_AIS"))
        PATH_JCOPE = path_jcope = json.loads(self.config_get_set_param(AREA, "PATH_JCOPE"))
    
    def get_shipLog_param(self):
        self.get_kalman_param()

