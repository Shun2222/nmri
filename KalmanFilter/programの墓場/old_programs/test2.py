import pickle as pkl
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

#year = 2015
#month = 9
#al = AISLoader(year, month)
#al.load_path()
is_exist =  pkl.load(open(osp.join('./data/cur_tf.pkl'), 'rb'))
ndata = pkl.load(open(osp.join('./data/cur_ndata.pkl'), 'rb'))

is_exist_each_day = [np.sum(is_exist[i*23:(i+1)*23], axis=1) for i in range(len(is_exist)//23)]
y = is_exist_each_day 
print(y)
#x = [np.arange(len(y[0])) for _ in range(len(y))]
x = np.arange(len(y[0])) 

y = np.sum(is_exist, axis=0)
x = np.arange(len(y)) 
plt.bar(x, y)
plt.title('num data each kuroshio index 9/1-9/30')
plt.xlabel('Index')
plt.ylabel('ndata')
plt.savefig('Images/ndata-eachIdx.png')
plt.close()
