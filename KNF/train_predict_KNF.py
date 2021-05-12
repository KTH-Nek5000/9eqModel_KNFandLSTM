"""
train_predict_KNF.py
---------------------
This code train a KNF model and uses it to predict
time series based on a given input seed and reproduce the long-term statistics.

Requires:
    train_data.mat - a file containing a time series used for training. This file
    contains a 3D array of size (1, nTP, 9) where
    nTP - number of time points
    test_data.mat - a file containing testing time series that were not used for 
    training. This file contains a 3D array of size (nTS, nTP, 9) where
    nTS - number of time series
    nTP - number of time points

Creates:
    ts_model_name.npz - file containing the reference and the predicted time series
    err_statistics_model_name.txt - file containing the error in the reproduction 
    of the long-term statistics
    err_coefs_model_name.txt - file containing the error in the instantaneous 
    predictions
    stats_model_name.txt - file containing the reference and the reconstructed
    long-term statistics
    
The code has been used for the results in:
    "Recurrent neural networks and Koopman-based frameworks for temporal
    predictions in a low-order model of turbulence"
    Hamidreza Eivazi, Luca Guastoni, Philipp Schlatter, Hossein Azizpour, 
    Ricardo Vinuesa
    International Journal of Heat and Fluid Flow (accepted)
    https://arxiv.org/abs/2005.02762

"""
import sys
sys.path.insert(0, '../Utilities/')

import numpy as np
from scipy.io import loadmat
from KNFs import KNF
from statistics import stats

from joblib import Parallel, delayed
import multiprocessing

direc = './../Datasets/'
file_train = direc + 'train_data.mat'
data_train = loadmat(file_train)['data']

n_train = 10000
q = 5
eta = 0

model_name = f'KNF_{n_train}_{q}'


train = data_train[0, :n_train, :]
train = train + np.random.normal(0.0, eta, np.shape(train))


knf = KNF(train, q, r_opt = False, ep = 1e-5, er = 1e-5, 
        polyorder = 4, sin_cos = 0, cons = True, excludes = [],
        sparse = True, threshold = 0.05);
knf.fit()


file_test = direc + 'test_data.mat'
data_test = loadmat(file_test)['data']

n_tst = 500
npred = 4000 - 1

def loop(j):
    true = data_test[j]
    x0 = true[:q]
    pred = knf.predict(x0, npred)
    pred = np.concatenate((true[0:1], pred))
    return pred, true


num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores, verbose = 2)(delayed(loop)(j) for j in range(n_tst))
results = np.array([i for i in results if i != None])

pred = results[:, 0].reshape(-1, 9)
true = results[:, 1].reshape(-1, 9)

del results

fname = f'./../Predictions/ts_{model_name}'
np.savez_compressed(fname, pred = pred, true = true)

Erru, Erru2 = stats(true, pred, model_name)
