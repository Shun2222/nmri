import numpy as np
import os.path as osp
import pickle as pkl

_Nslice = 20
isExistShipLog = isTarget = pkl.load(open(osp.join('./data/isExistShipLog.pkl'), 'rb'))
_Targets = np.where(isTarget)[0]
#for r in range(len(cur1)//_Nslice + 1):
for r in range(len(_Targets)//_Nslice+1):
    day = 1

    _N = _Nslice
    TF = np.array([False for _ in range(9808)])
    targets = _Targets[r*_N:(r+1)*_N]
    for target in targets:
        TF[target] = True
    print(len(targets))
    if np.sum(TF)!=_Nslice:
        _N = np.sum(TF)
        m = _M = 1
        _NM = _N * _M
        _2NM = 2 * _NM
        print(_N)