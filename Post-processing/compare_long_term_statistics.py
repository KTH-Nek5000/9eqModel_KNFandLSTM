"""
compare_long_term_statistics.py
---------------------
This file compares the performance of KNF, HDMD, and LSTM models in the 
reproduction of the long-term statistics by producing Figures 5 of the paper.

Requires:
    stats_KNF_model_name.npz - reproduction of the long-term statistics by KNF.
    stats_HDMD_model_name.npz - reproduction of the long-term statistics by HDMD.
    stats_LSTM_model_name.npz - reproduction of the long-term statistics by LSTM.
    
    Each of these files contain a 2-D array of size (nGP, 6) where
    nGP - number of grid points
    Rows represent:
        u_ref
        u_pred
        uu_ref
        uu_pred
        uv_ref
        uv_pred

Creates:
    Figures 5 of the paper.
    
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
from scipy.interpolate import make_interp_spline

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 14)
plt.rc('legend', fontsize = 12, handletextpad=0.3)              
plt.rc('xtick', labelsize = 14)             
plt.rc('ytick', labelsize = 14)

direc = './../Predictions/'
models = ['KNF_10000_5', 'HDMD_10000_5', 'LSTM1_t10000']

knf = np.loadtxt(direc + 'stats_' + models[0] + '.txt') 
hdmd = np.loadtxt(direc + 'stats_' + models[1] + '.txt')
lstm = np.loadtxt(direc + 'stats_' + models[2] + '.txt')

u_ref = knf[0]; uu_ref = knf[2]; uv_ref = knf[4]
u_knf = knf[1]; uu_knf = knf[3]; uv_knf = knf[5]
u_hdmd = hdmd[1]; uu_hdmd = hdmd[3]; uv_hdmd = hdmd[5]
u_lstm = lstm[1];  uu_lstm = lstm[3]; uv_lstm = lstm[5]


Np = 21
Y = np.linspace(-1, 1, Np)
Yn = np.linspace(-1, 1, 51)

def spl(x):
    spl1 = make_interp_spline(Y, x, k=3)
    y = spl1(Yn)
    return y

u_ref = spl(u_ref)
u_knf = spl(u_knf)
u_lstm = spl(u_lstm)
u_hdmd = spl(u_hdmd)

uu_ref = spl(uu_ref)
uu_knf = spl(uu_knf)
uu_lstm = spl(uu_lstm)
uu_hdmd = spl(uu_hdmd)

uv_ref = spl(uv_ref)
uv_knf = spl(uv_knf)
uv_lstm = spl(uv_lstm)
uv_hdmd = spl(uv_hdmd)

lsiz = 18
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = False, sharey = True, figsize = (10, 4))

ax1.plot(u_ref, Yn, color = 'k', linestyle = '--', label = 'Reference', linewidth = 1)
ax1.plot(u_knf, Yn, color = 'tab:orange', linestyle = '-', label = 'KNF', linewidth = 1)
ax1.plot(u_lstm, Yn, color = 'tab:blue', label = 'LSTM', linewidth = 1)
ax1.plot(u_hdmd, Yn, color = 'tab:green', label = 'HDMD', linewidth = 1)

ax1.set_xlabel('$\overline{u}$', fontsize = lsiz)
ax1.set_ylabel('$y$', fontsize = lsiz)
ax1.set_xlim(-0.5, 0.5)
ax1.set_ylim(-1, 1)
ax1.set_yticks(np.linspace(1, -1, 5))
ax1.legend(frameon = False, loc = 0)
	
ax2.plot(uu_ref, Yn, color = 'k', linestyle = '--', linewidth = 1)
ax2.plot(uu_knf, Yn, color = 'tab:orange', linestyle = '-', linewidth = 1)
ax2.plot(uu_lstm, Yn, color = 'tab:blue', linewidth = 1)
ax2.plot(uu_hdmd, Yn, color = 'tab:green', linewidth = 1)

ax2.set_xlabel('$\overline{u^{\prime 2}}$', fontsize = lsiz)
ax2.set_xlim(-0.01, 0.06)

ax3.plot(uv_ref, Yn, color = 'k', linestyle = '--', linewidth = 1)
ax3.plot(uv_knf, Yn, color = 'tab:orange', linestyle = '-', linewidth = 1)
ax3.plot(uv_lstm, Yn, color = 'tab:blue', linewidth = 1)
ax3.plot(uv_hdmd, Yn, color = 'tab:green', linewidth = 1)

ax3.set_xlabel('$\overline{u^{\prime}v^{\prime}}$', fontsize = lsiz)
ax3.set_xlim(-0.007, 0.001)
plt.subplots_adjust(wspace = 0.3)


# plt.savefig('statistics.pdf', bbox_inches='tight')