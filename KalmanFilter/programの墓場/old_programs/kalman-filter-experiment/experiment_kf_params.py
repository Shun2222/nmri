import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.gh import GHFilter
from numpy.random import randn
import seaborn as sns
import tqdm
import pickle as pkl

deg_per_mesh = 1/36
ais_lat_max = 1800
ais_lat_min = 750
ais_lon_max = 5401
ais_lon_min = 4211
lat_range = [20-1/36, 50+1/36]
lon_range = [117-1/36, 150+1/36]
map_size_ais = [1050, 1191]
map_size_jcope = [1082, 1191]
pool_size = 3 
map_pooled_size = (int(map_size_ais[0]/pool_size), int(map_size_ais[1]/pool_size))

# Only East Area of KUROSHIO
# kurosio_lat_range1 = [30, 32]
# kurosio_lat_range2 = [34, 36]
# kurosio_lon_range = [136, 142]
kurosio_lat_range1 = [30.0, 32.0] # x00 x01 #CHANGED
kurosio_lat_range2 = [30.0, 34.0] # x10 x11
kurosio_lon_range = [131, 136] # y0 y1
line1 = [[kurosio_lon_range[0], kurosio_lat_range1[0]], [kurosio_lon_range[1], kurosio_lat_range2[0]]]
line2 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]
line4 = [[kurosio_lon_range[0], kurosio_lat_range1[1]], [kurosio_lon_range[0], kurosio_lat_range2[0]]]
line3 = [[kurosio_lon_range[1], kurosio_lat_range2[0]], [kurosio_lon_range[1], kurosio_lat_range2[1]]]

kurosio_latidx_range1 = [int((kurosio_lat_range1[0]-lat_range[0])/deg_per_mesh), int((kurosio_lat_range1[1]-lat_range[0])/deg_per_mesh)]
kurosio_latidx_range2 = [int((kurosio_lat_range2[0]-lat_range[0])/deg_per_mesh), int((kurosio_lat_range2[1]-lat_range[0])/deg_per_mesh)]
kurosio_lonidx_range = [int((kurosio_lon_range[0]-lon_range[0])/deg_per_mesh), int((kurosio_lon_range[1]-lon_range[0])/deg_per_mesh)]