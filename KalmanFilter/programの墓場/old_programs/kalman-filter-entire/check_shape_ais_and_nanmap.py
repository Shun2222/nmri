import pandas as pd

aisx = pd.read_csv('aisx.csv').values
print(f'aisx {aisx.shape}')

nan_map = pd.read_csv('nan_map_pooled6.csv').values
print(f'nan_map {nan_map.shape}')

data = aisx * nan_map
df = pd.DataFrame(data)
path = 'aisx-nanmap.csv'
df.to_csv(path, header=False, index=False)
print(f'saved as {path}')