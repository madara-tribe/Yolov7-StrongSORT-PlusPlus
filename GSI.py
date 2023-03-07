import os
import numpy as np
from os.path import join
from collections import defaultdict
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# linear interpolation
def LinearInterpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # sorted by ID and frames
    output_ = input_.copy()
    '''linear interpolation'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # same ID
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # interpolation at each frames
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:  # other ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_

# Gaussian smoothing
def GaussianSmooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])
    for i, id_ in enumerate(ids):
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)[:, 0]
        gpr.fit(t, y)
        yy = gpr.predict(t)[:, 0]
        gpr.fit(t, w)
        ww = gpr.predict(t)[:, 0]
        gpr.fit(t, h)
        hh = gpr.predict(t)[:, 0]
        output_.extend([
            [t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1 , -1] for i in range(len(t))
        ])
    return output_

# GSI
def GSInterpolation(path_in, path_out, interval, tau, save=None):
    input_ = np.loadtxt(path_in, delimiter=',')
    li = LinearInterpolation(input_, interval)
    gsi = GaussianSmooth(li, tau)
    gsi = np.array(gsi).astype(int)
    print('gsi shape', gsi.shape)
    if save:
        gsi = np.array(gsi).astype(int)
        gsi = gsi[np.lexsort([gsi[:, 1], gsi[:, 0]])]
        np.savetxt(path_out, gsi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        

