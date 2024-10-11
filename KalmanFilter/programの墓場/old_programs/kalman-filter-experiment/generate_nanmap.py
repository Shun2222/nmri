import pandas as pd
from experiment_utils import *
from experiment_kf_params import *

# nan map data
nan_map_pooled3 = average_pooling(nan_map[:map_size_ais[0]], pool_size=(pool_size, pool_size))

# save
df = pd.DataFrame(nan_map_pooled3)
path = 'nan_map_pooled3.csv'
df.to_csv(path, header=False, index=False)
print(f'saved as {path}')