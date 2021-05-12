"""
predict_LSTM.py
---------------------
This code uses a trained LSTM model to predict
time series based on a given input seed and reproduce the long-term statistics.

Requires:
    LSTM#_#.h5 - a trained LSTM model
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
from tensorflow.keras import models
from scipy.io import loadmat
from lstm_pred_func import lstm_pred
from statistics import stats



model_name = 'LSTM1_t10000'
lstm = models.load_model(model_name + '.h5')
print(lstm.summary())


p = 10
nts = 500

direc = './../Datasets/'
file_test = direc + './test_data.mat'

true = loadmat(file_test)['data']
ntp = true.shape[1]
pred_steps = ntp - p

pred = np.zeros(true.shape)
for i in range(nts):
    pred[i] = lstm_pred(lstm, true[i], p, pred_steps)
    print(i)
	

true = true.reshape((-1, 9))	
pred = pred.reshape((-1, 9))

fname = f'./../Predictions/ts_{model_name}'
np.savez_compressed(fname, pred = pred, true = true)

Erru, Erru2 = stats(true, pred, model_name)
