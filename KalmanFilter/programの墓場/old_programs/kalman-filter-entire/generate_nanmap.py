import pandas as pd
from entire_utils import *
from entire_kf_params import *

# nan map data
nan_map_pooled6 = average_pooling(nan_map[:map_size_ais[0]], pool_size=(pool_size, pool_size))

# save
df = pd.DataFrame(nan_map_pooled6)
path = 'nan_map_pooled6.csv'
df.to_csv(path, header=False, index=False)
print(f'saved as {path}')