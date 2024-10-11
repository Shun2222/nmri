import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

num_data = 119 
save_path = r"E:/shunsukeE/result/kalman-enough_data/" 

def load_xzj(day, hour, s, thres=0.6, save_path=''):
    pathX = save_path + fr'saverX201509{day:02}{hour:02}-{s}.pkl'
    pathXCur = save_path + fr'saverXCur201509{day:02}{hour:02}-{s}.pkl'
    pathZ = save_path + fr'saverZ201509{day:02}{hour:02}-{s}.pkl'
    pathH = save_path + fr'saverH201509{day:02}{hour:02}-{s}.pkl'
    pathP = save_path + fr'saverP201509{day:02}{hour:02}-{s}.pkl'
    pathF = save_path + fr'saverF201509{day:02}{hour:02}-{s}.pkl'
    pathR = save_path + fr'saverR201509{day:02}{hour:02}-{s}.pkl'
    pathJCOPE = save_path + fr'saverJCOPE201509{day:02}{hour:02}-{s}.pkl'
    pathJCOPECur = save_path + fr'saverJCOPECur201509{day:02}{hour:02}-{s}.pkl'

    x = pkl.load(open(pathXCur, 'rb'))
    z = pkl.load(open(pathZ, 'rb'))
    jcope = pkl.load(open(pathJCOPECur, 'rb'))

    return x, z, jcope

def diff_xzj(x, z, j):
    thres = 0.6

    a = x-z
    print(f'kalman-ais')
    print(f'Mean: {np.mean(a)}')
    print(f'Under {thres} rate:{np.sum(a<thres)/np.sum(a==a)}')

    a = x-jcope
    print(f'kalman-jcope')
    print(f'Mean: {np.mean(a)}')
    print(f'Under {thres} rate:{np.sum(a<thres)/np.sum(a==a)}')

    a = z-jcope
    print(f'ais-jcope')
    print(f'Mean: {np.mean(a)}')
    print(f'Under {thres} rate:{np.sum(a<thres)/np.sum(a==a)}')
    

ais = []
kalman = []
jcope = []
#for s in range(1):
s = 0
for day in range(1, 30-1):
    for hour in range(0, 23): #694
        x, z, j = load_xzj(day, hour, s, save_path=save_path)
        ais.append(z)
        kalman.append(x)
        jcope.append(j)
        print(len(z))
diff_xzj(np.array(kalman), np.array(ais), np.array(jcope))

ais = np.array(ais).reshape(len(ais), len(ais[0]))
kalman = np.array(kalman).reshape(len(kalman), len(kalman[0]))
jcope = np.array(jcope).reshape(len(jcope), len(jcope[0]))

n = len(ais[0])#20
k = int(n/2)
print(n)
x = np.arange(len(ais))
plt.plot(x, jcope.T[0], color='b', label='JCOPE')
plt.plot(x, ais.T[0], color='r', label='AIS')
plt.plot(x, kalman.T[0], color='g', label='Kalman')
for i in range(1, k):
    plt.plot(x, jcope.T[i], color='b')
for i in range(1, k):
    plt.plot(x, ais.T[i], color='r')
for i in range(1, k):
    plt.plot(x, kalman.T[i], color='g')

plt.xlabel('Time(hour)')
plt.ylabel('v1')
#plt.legend(bbox_to_anchor=(0.5, 0.0), ncol=3)
plt.legend()
plt.savefig('Images/kalman-Ptest_x_ST.png')
plt.show()

plt.plot(x, jcope.T[k], color='b', label='JCOPE')
plt.plot(x, ais.T[k], color='r', label='AIS')
plt.plot(x, kalman.T[k], color='g', label='Kalman')
for i in range(1, k):
    plt.plot(x, jcope.T[k+i], color='b')
for i in range(1, k):
    plt.plot(x, ais.T[k+i], color='r')
for i in range(1, k):
    plt.plot(x, kalman.T[k+i], color='g')

plt.xlabel('Time(hour)')
plt.ylabel('v2')
#plt.legend(bbox_to_anchor=(0.5, 0.0), ncol=3)
plt.legend()
plt.savefig('Images/kalman-Ptest_ST_y.png')
plt.show()
