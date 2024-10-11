import numpy as np
import pickle as pkl

class KalmanFilter:
    def __init__(self, logger, _N, _M, x, z, notNan, F, H, Q, R):
        self.logger = logger
        self._N = _N
        self._M = _M
        self._NM = _N*_M
        self.Nt = int(len(z))
        self.Mt = 1 
        self.NMt = self.Nt * self.Mt
        self.x = x
        self.z = z 
        self.F = F
        self.H = H
        self.Q = Q 
        self.P = np.zeros((self.NMt+1, self.NMt+1)) 
        self.I = np.eye(self.NMt+1) 
        self.K = np.eye(self.NMt+1)
        self.R = np.zeros((self.NMt+1, self.NMt+1)) 
        self.notNan = notNan

    def predict(self, F=None, Q=None):
        if not F is None:
            self.F = F
        if not Q is None:
            self.Q = Q
        self.x = self.F @ self.x #x k|k-1
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, notNan, H=None, R=None):
        self.notNan = notNan
        notNan_ravel = notNan.ravel()
        self.Nt = int(len(z))
        self.Mt = 1 
        self.NMt = self.Nt * self.Mt
        self.I = np.eye(self.NMt+1)
        if not H is None:
            self.H = H
        if not R is None:
            self.R = R

        if self.Nt==1:
            self.K = None
            return

        PHT = self.P[notNan_ravel].T[notNan_ravel].T @ self.H.T
        S = self.H @ PHT + self.R

        SI = np.linalg.pinv(S.T)
        self.K = K = PHT @ SI

        y = z - self.H @ self.x[notNan]
        x[notNan] = x[notNan] + K @ y # x k|k

        I_KH = I - K @ H
        # TODO ちょっと知ってるのと違う，式的にRを写像してPに加えてる，Rの誤差も考慮するようにしてる?
        PnotNan = I_KH @ P[notNan_ravel].T[notNan_ravel].T @ I_KH.T + K @ R @ K.T
        PnotNan_prev = P[notNan_ravel]
        for i in range(len(PnotNan)):
            PnotNan_prev[i][notNan_ravel] = PnotNan[i]
        self.P[notNan_ravel] = PnotNan

    def logger(self, jcope):
        def mean_diff(x, y):
            a = x-y
            diff = np.mean(np.abs(a))
            return diff
        self.logger.record_tabular(f"dtidx", dtidx)
        self.logger.record_tabular(f"AIS-JCOPE", mean_diff(self.z, self.H@jcope[self.notNan]))
        self.logger.record_tabular(f"Kalman-JCOPE", mean_diff(self.H@x, self.H@jcope[self.notNan]))
        self.logger.record_tabular(f"AIS-Kalman", mean_diff(self.z, self.H@self.x[self.notNan]))
        self.logger.record_tabular(f"Available z", np.sum(self.notNan))
        self.logger.dump_tabular()

    def save(self, jcope, fname):
        save_dir = self.logger.get_dir()
        # 保存
        with open(f'{save_dir}/saverX.pkl', 'wb') as f:
            pkl.dump(self.x, f) #KALMAN
        with open(f'{save_dir}/saverZ{fname}.pkl', 'wb') as f:
            pkl.dump(self.z, f) #AIS
        with open(f'{save_dir}/saverJCOPE{fname}.pkl', 'wb') as f:
            pkl.dump(jcope, f) #JCOPE
        with open(f'{save_dir}/saverP{fname}.pkl', 'wb') as f:
            pkl.dump(self.P, f)
        with open(f'{save_dir}/saverR{fname}.pkl', 'wb') as f:
            pkl.dump(self.R, f)
        with open(f'{save_dir}/saverF{fname}.pkl', 'wb') as f:
            pkl.dump(self.F, f)
        with open(f'{save_dir}/saverH{fname}.pkl', 'wb') as f:
            pkl.dump(self.H, f)
        with open(f'{save_dir}/saverK{fname}.pkl', 'wb') as f:
            pkl.dump(self.K, f)
        with open(f'{save_dir}/saverJCOPECur{fname}.pkl', 'wb') as f:
            if np.sum(self.notNan)!=1:
                HJcope = self.H @ jcope[self.notNan]
            else:
                HJcope = None
            pkl.dump(HJcope, f)
        with open(f'{save_dir}/saverXCur{fname}.pkl', 'wb') as f:
            if np.sum(self.notNan)!=1:
                Hx = self.H@self.x[self.notNan]
            else:
                Hx = None
            pkl.dump(Hx, f)
