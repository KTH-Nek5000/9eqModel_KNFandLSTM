#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:15:33 2020

@author: luca
"""

import numpy as np
import math
#import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import os

cur_path = os.path.dirname(__file__)
ds_ode_path = '../../Datasets/perturbed_1e-06/'
ds_lstm_path = '../../Datasets/perturbed_series_lstm_1e-06/'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=55)
plt.rc('font', size=25)
plt.rc('legend', fontsize=8)               # Make the legend/label fonts 
plt.rc('xtick', labelsize=55)             # a little smaller
plt.rc('ytick', labelsize=55)

T=4000
nSeries = 10
t_perturb = np.array([500])
t_poly = ([964])#, 1913, 1935, 2800])         #average values 500

test = np.ndarray((len(t_perturb),9,T),float)
pert = np.ndarray((len(t_perturb),9,T),float)
lyap = np.ndarray((len(t_perturb),T),float)
log_lyap = np.ndarray((len(t_perturb),T),float)

lyap[:,:] = 0.0

fig = plt.figure()
ax = fig.add_subplot(111)

#%% Lyapunov exponents for the ODE
for j in range(0,len(t_perturb)):
    for jj in range(0,nSeries):
        matfile = sio.loadmat(ds_ode_path+'pert_'+str(jj+1)+'_'+str(t_perturb[j])+'.mat')
        
        test[j,:,:] = matfile['testSeq']
        pert[j,:,:] = matfile['pertSeq']
        
        diff_t = np.linalg.norm(test[j,:,:]-pert[j,:,:],axis=0)
        diff_i = np.copy(diff_t[t_perturb[j]+1])
        
        for i in range(0, T):       
            if i < t_perturb[j]+1:
                lyap[j,i] = np.nan    
            else:
                lyap[j,i] += (diff_t[i])/nSeries
                
    log_lyap[j,:] = np.log(lyap[j,:])            
    p = np.polyfit(np.arange(t_perturb[j]+1, t_poly[j]+1, 1.0), \
                    log_lyap[j,t_perturb[j]+1:t_poly[j]+1], 1)

    p_lin_ode = ax.plot(np.arange(t_perturb[j]+1, t_poly[j]+1, 1.0), \
              np.exp(p[1])*np.exp(p[0]*np.arange(t_perturb[j]+1, t_poly[j]+1, 1.0)), \
              color='C1', linestyle='-.', linewidth=1.5)
    plt.text(1200, 10**(-3), '$\lambda_{ODE} =\ $'+str(round(p[0],4)), color='C1', fontsize=50)   
    p_ode = ax.plot(lyap[j,:], 'C1', linewidth=3.0)

    
 
#%% Lyapunov exponents for the LSTM
pred = np.ndarray((len(t_perturb),9,T),float)
test1 = np.ndarray((len(t_perturb),9,T),float)

tpred = np.ndarray((len(t_perturb),9,T),float)
lyap2 = np.ndarray((len(t_perturb),T),float)
log_lyap2 = np.ndarray((len(t_perturb),T),float)

lyap2[:,:] = 0.0

for j in range(0,len(t_perturb)):
    for jj in range(0,nSeries):
        matfile = sio.loadmat(ds_lstm_path+'pert_'+str(jj+1)+'_'+str(t_perturb[j])+'_lstm.mat')
        pred[j,:,:] = np.transpose(matfile['pertSeq_lstm'])
        test1[j,:,:] = np.transpose(matfile['testSeq_lstm'])
        
        diff_t = np.linalg.norm(test1[j,:,:]-pred[j,:,:],axis=0)
        diff_i = np.copy(diff_t[t_perturb[j]+1])
        
        for i in range(0, T):       
            if i < t_perturb[j]+1:
                lyap2[j,i] = np.nan    
            else:
                lyap2[j,i] += (diff_t[i])/nSeries
                
    log_lyap2[j,:] = np.log(lyap2[j,:])            
    p_lstm = np.polyfit(np.arange(t_perturb[j]+1, t_poly[j]+1, 1.0), \
                        log_lyap2[j,t_perturb[j]+1:t_poly[j]+1], 1)
        
    p_lin_lstm = ax.plot(np.arange(t_perturb[j]+1, t_poly[j]+1, 1.0), \
                         np.exp(p_lstm[1])*np.exp(p_lstm[0]*np.arange(t_perturb[j]+1, t_poly[j]+1, 1.0)), \
                         color='C0', linestyle='-.', linewidth=1.5)
        
    plt.text(1200, 10**(-4), '$\lambda_{LSTM} =\ $'+str(round(p_lstm[0],4)), color='C0', fontsize=50)   
    plt.plot(lyap2[j,:], 'C0', linewidth=3)
    

ax.set(xlim=(-50,2000), yscale='log', xlabel='$\delta t$', ylabel='$|\delta \mathbf{A}(t)|$')



