"""
compare_short_term_predictions.py
---------------------
This file compares the performance of KNF, HDMD, and LSTM models in the 
short-term predictions by producing Figures 2 and 3 of the paper.

Requires:
    ts_KNF_model_name.npz - KNF predictions. 
    ts_HDMD_model_name.npz - HDMD predictions.
    ts_LSTM_model_name.npz - LSTM predictions.
    
    Each of these files contain a 2-D array of size (nTS * nTP, 9) where
    nTS - number of time series
    nTP - number of time points

Creates:
    Figures 2 and 3 of the paper.
    
The code has been used for the results in:
    "Recurrent neural networks and Koopman-based frameworks for temporal
    predictions in a low-order model of turbulence"
    Hamidreza Eivazi, Luca Guastoni, Philipp Schlatter, Hossein Azizpour, 
    Ricardo Vinuesa
    International Journal of Heat and Fluid Flow (accepted)
    https://arxiv.org/abs/2005.02762
    
"""
import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 14)
plt.rc('legend', fontsize = 12)              
plt.rc('xtick', labelsize = 14)             
plt.rc('ytick', labelsize = 14)


direc = './../Predictions/'
models = ['ts_KNF_10000_5', 'ts_HDMD_10000_5', 'ts_LSTM1_t10000']


def cal(name):
    data = np.load(direc+name+'.npz')
    pred = data['pred'].reshape((-1, 4000, 9))
    true = data['true'].reshape((-1, 4000, 9))

    e = np.abs(pred - true)
    e = np.linalg.norm(e, axis = 2); e = e.T / np.mean(np.linalg.norm(true, axis = 2), axis = 1); e = e.T
    et = e
    e = np.mean(e, axis = 0)
    return pred, true, e, et

pred_KNF, true, error_KNF, et = cal(models[0])
pred_HDMD, _, error_HDMD, _ = cal(models[1])
pred_LSTM, _, error_LSTM, _ = cal(models[2])

divp = []
for i in range(500):
    divp.append([j for j in range(3999) if np.all(et[i, :j]) < 0.3 and et[i, j+1] > 0.3][0])

#%%
plt.figure(figsize = (8, 4))
plt.subplot(2, 1, 1)
ns = np.argmax(divp)
maxdivp = divp[ns]
n = 0

plt.plot(pred_KNF[ns, :, n], c = 'tab:orange', label = 'KNF', linewidth = 1)
plt.plot(pred_HDMD[ns, :, n], c = 'tab:green', label = 'HDMD', linewidth = 1)
plt.plot(pred_LSTM[ns, :, n], c = 'tab:blue', label = 'LSTM', linewidth = 1)
plt.plot(true[ns, :, n], c = 'k', label = 'Reference', linestyle = '--', linewidth = 1)
plt.plot([maxdivp, maxdivp], [-1, 1], c = 'gray', linestyle = '--', linewidth = 1)

plt.ylabel(f'$a_{n+1}$', fontsize = 20)
plt.xticks(np.arange(0, 1001, 200), [])
plt.xlim(0, 1000)
plt.ylim(0, 0.6)
plt.legend(frameon = False, loc = 0, ncol = 4)

plt.subplot(2, 1, 2)
ns = np.argmin(divp)
mindivp = divp[ns]
n = 0

plt.plot(pred_KNF[ns, :, n], c = 'tab:orange', label = 'KNF', linewidth = 1)
plt.plot(pred_HDMD[ns, :, n], c = 'tab:green', label = 'HDMD', linewidth = 1)
plt.plot(pred_LSTM[ns, :, n], c = 'tab:blue', label = 'LSTM', linewidth = 1)
plt.plot(true[ns, :, n], c = 'k', label = 'Reference', linestyle = '--', linewidth = 1)
plt.plot([mindivp, mindivp], [-1, 1], c = 'gray', linestyle = '--', linewidth = 1)

plt.ylabel(f'$a_{n+1}$', fontsize = 20)
plt.xlabel('$t$', fontsize = 20)
plt.xticks(np.arange(0, 1001, 200))
plt.xlim(0, 1000)
plt.ylim(0, 0.8)

# plt.savefig('./Coefs.pdf', bbox_inches='tight')
#%%
s = 2000
plt.figure(figsize = (8, 2))
plt.plot(error_KNF[:s], label = 'KNF', c = 'tab:orange', linewidth = 1)
plt.plot(error_HDMD[:s], label = 'HDMD', c = 'tab:green', linewidth = 1)
plt.plot(error_LSTM[:s], label = 'LSTM', c = 'tab:blue', linewidth = 1)
plt.plot(np.arange(s), np.zeros(s) + 0.3, c = 'gray', linestyle = '--', linewidth = 1)

plt.xlabel('$t$', fontsize = 20)
plt.ylabel('$\epsilon$', fontsize = 20)
plt.legend(frameon = False, loc = 0, ncol = 3)
plt.xlim(0, s)
plt.ylim(-0.1, 1.4)

# plt.savefig('./short_term_error.pdf', bbox_inches='tight')
