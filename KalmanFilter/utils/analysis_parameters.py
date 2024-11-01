import ast
import json 
from utils.parameter_manager import ParameterManager 
import utils.logger as logger

lat_range = [20-1/36, 50+1/36]
lon_range = [117-1/36, 150+1/36]
map_size_ais = [1050, 1191]
map_size_jcope = [1082, 1191]
deg_per_mesh = 1/36

# get analysis parameters
project = "analysis"
paramManager = ParameterManager(project)
params = paramManager.get_param()

# Analysis params
path_log = json.loads(params["PATH_LOG"])
path_ship = json.loads(params["PATH_SHIP"])
DAY = json.loads(params["DAY"])
HOUR = json.loads(params["HOUR"])
MAX_DAY = json.loads(params["MAX_DAY"])
MAX_HOUR = json.loads(params["MAX_HOUR"])
AIS_JCOPE_TEST = shipLog_analysis = json.loads(params["AIS_JCOPE_TEST"])
VISUALIZE = shipLog_analysis = json.loads(params["VISUALIZE"])
EVALUATION = shipLog_analysis = json.loads(params["EVALUATION"])

# AREA params
AREA = params["AREA"]
USE_AIS_REMOVED_BAD_MMSI = use_ais_removed_bad_mmsi = params["USE_AIS_REMOVED_BAD_MMSI"]
pool_size = int(params["POOL_SIZE"])
kurosio_lat_range1 = ast.literal_eval(params["KUROSIO_LAT_RANGE1"])
kurosio_lat_range2 = ast.literal_eval(params["KUROSIO_LAT_RANGE2"])
kurosio_lon_range = ast.literal_eval(params["KUROSIO_LON_RANGE"])
PATH_AIS = path_ais = json.loads(params["PATH_AIS"])
PATH_JCOPE = path_jcope = json.loads(params["PATH_JCOPE"])

map_pooled_size = (int(map_size_ais[0]/pool_size), int(map_size_ais[1]/pool_size))

line1 = [[kurosio_lon_range[0], kurosio_lat_range1[0]], [kurosio_lon_range[1], kurosio_lat_range2[0]]]
line2 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]
line3 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[0], kurosio_lat_range2[0]]]
line4 = [[kurosio_lon_range[1], kurosio_lat_range2[0]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]

kurosio_latidx_range1 = [int((kurosio_lat_range1[0]-lat_range[0])/deg_per_mesh), int((kurosio_lat_range1[1]-lat_range[0])/deg_per_mesh)]
kurosio_latidx_range2 = [int((kurosio_lat_range2[0]-lat_range[0])/deg_per_mesh), int((kurosio_lat_range2[1]-lat_range[0])/deg_per_mesh)]
kurosio_lonidx_range = [int((kurosio_lon_range[0]-lon_range[0])/deg_per_mesh), int((kurosio_lon_range[1]-lon_range[0])/deg_per_mesh)]

def dump_params():
    for key in params:
        logger.record_tabular(key, params[key])
    logger.dump_tabular()