"""
statistics.py
---------------------
This code contains a function that computes the long-term statistics from 
the given reference and predicted time series

Requires:
    true: 3-D array 
            The reference time series with the size of (nTS, nTP, 9) where
            nTS - number of time series
            nTP - number of time points
    
    pred: 3-D array
            The predicted time series with the same size as true
            
    model_name: string
            Name of the model

Creates:
    err_statistics_model_name.txt - file containing the error in the reproduction 
    of the long-term statistics
    err_coefs_model_name.txt - file containing the error in the instantaneous 
    predictions
    stats_model_name.txt - file containing the reference and the reconstructed
    long-term statistics
    
Returns:
    Erru, Erru2: float
            The errors in the long-term statistics
    
The code has been used for the results in:
    "Recurrent neural networks and Koopman-based frameworks for temporal
    predictions in a low-order model of turbulence"
    Hamidreza Eivazi, Luca Guastoni, Philipp Schlatter, Hossein Azizpour, 
    Ricardo Vinuesa
    International Journal of Heat and Fluid Flow (accepted)
    https://arxiv.org/abs/2005.02762

"""
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def stats(true, pred, model_name):
    
    Lx = 4 * np.pi
    Ly = 2
    Lz = 2 * np.pi
    
    al = 2 * np.pi / Lx
    b = np.pi / Ly
    g = 2 * np.pi / Lz
    
    Np = 21
    Np2 = 21
    X = np.linspace(0, Lx, Np2)
    Y = np.linspace(-1, 1, Np)
    Z = np.linspace(0, Lz, Np2)
    
    
    def modes(x, y, z, al, b, g):
    	u = np.zeros((3, 9))
    	
    	u[:, 0] = np.array([np.sqrt(2) * np.sin(np.pi * y / 2), 0, 0])
    	
    	u[:, 1] = np.array([4 / np.sqrt(3) * np.cos(np.pi * y / 2)**2 * np.cos(g * z), 0, 0])
    	
    	u[:, 2] = 2 / np.sqrt(4 * g**2 + np.pi**2) * np.array([0, 2 * g * np.cos(np.pi *y / 2) * np.cos(g * z), np.pi * np.sin(np.pi * y / 2) * np.sin(g * z)])
    	
    	u[:, 3] = np.array([0, 0, 4 / np.sqrt(3) * np.cos(al * x) * np.cos(np.pi * y / 2)**2])
    	
    	u[:, 4] = np.array([0, 0, 2 * np.sin(al * x) * np.sin(np.pi * y / 2)])
    	
    	u[:, 5] = 4 * np.sqrt(2 / (3 * (al**2 + g**2))) * np.array([ -g * np.cos(al * x) * np.cos(np.pi * y / 2)**2 * np.sin(g * z), 0, al * np.sin(al * x) * np.cos(np.pi * y / 2)**2 * np.cos(g * z)])
    	
    	u[:, 6] = 2 * np.sqrt(2 / (al**2 + g**2)) * np.array([ g * np.sin(al * x) * np.sin(np.pi * y/2) * np.sin(g*z), 0, al * np.cos(al*x) * np.sin(np.pi*y/2) * np.cos(g*z)])
    	
    	n8 = 2 * np.sqrt(2 / ((al**2 + g**2) * (4*al**2 + 4*g**2 + np.pi**2)))
    	u[:, 7] = n8 * np.array([np.pi * al * np.sin(al * x) * np.sin(np.pi * y/2) * np.sin(g * z), 2 * (al**2 + g**2) * np.cos(al * x) * np.cos(np.pi * y / 2) * np.sin(g * z), -np.pi * g * np.cos(al * x) * np.sin(np.pi * y / 2) * np.cos(g * z)])
    	
    	u[:, 8] = np.array([np.sqrt(2) * np.sin(3 * np.pi * y / 2), 0, 0])
    	return u
    

    ux = np.zeros((Np2, Np2, Np)); uxp = np.zeros((Np2, Np2, Np))
    uux = np.zeros((Np2, Np2, Np)); uuxp = np.zeros((Np2, Np2, Np))
    uv = np.zeros((Np2, Np2, Np)); uvp = np.zeros((Np2, Np2, Np))
    M = np.zeros((Np2, Np, Np2, 3, 9))

    
    for i in range(Np2):
        for j in range(Np):
            for k in range(Np2):
                mods = modes(X[i], Y[j], Z[k], al, b, g)
                M[i, j, k] = mods
                

    def loop_u(i):
        for k in range(Np2):
            for j in range(Np):
                ut = M[i, j, k] @ true.T
                up = M[i, j, k] @ pred.T
                
                Uit = ut[0].mean()
                Uip = up[0].mean()
                Ujt = ut[1].mean()
                Ujp = up[1].mean()
                
                ux[i, k, j] = Uit
                uxp[i, k, j] = Uip
             	
                uux[i, k, j] = np.mean(ut[0]**2) - Uit**2
                uuxp[i, k, j] = np.mean(up[0]**2) - Uip**2
            
                uv[i, k, j] = np.mean(ut[0] * ut[1]) - Uit * Ujt
                uvp[i, k, j] = np.mean(up[0] * up[1]) - Uip * Ujp
        return ux[i], uxp[i], uux[i], uuxp[i], uv[i], uvp[i]
    
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores, verbose = 11)(delayed(loop_u)(i) for i in range(Np2))
    
    for i in range(Np2):
        ux[i] = res[i][0]
        uxp[i] = res[i][1]
        uux[i] = res[i][2]
        uuxp[i] = res[i][3]
        uv[i] = res[i][4]
        uvp[i] = res[i][5]
        
    del res
          
    ux = ux.mean(axis = 0)
    uxp = uxp.mean(axis = 0)
    ux = ux.mean(axis = 0)
    uxp = uxp.mean(axis = 0)
    
    uux = uux.mean(axis = 0)
    uuxp = uuxp.mean(axis = 0)
    uux = uux.mean(axis = 0)
    uuxp = uuxp.mean(axis = 0)
    
    uv = uv.mean(axis = 0)
    uvp = uvp.mean(axis = 0)
    uv = uv.mean(axis = 0)
    uvp = uvp.mean(axis = 0)
    

    Erru = 1/(np.max(ux))* np.mean(np.abs(ux - uxp))
    Erru2 = 1/(np.max(uux))* np.mean(np.abs(uux - uuxp))
    Err = np.array([Erru, Erru2])
    np.savetxt(f'./../Predictions/err_statistics_{model_name}.txt', Err)

    ErrCoefs = np.mean(np.abs(true - pred), axis = 0)
    
    np.savetxt(f'./../Predictions/err_coefs_{model_name}.txt', ErrCoefs)
    

    stats = np.vstack((ux, uxp, uux, uuxp, uv, uvp))
    np.savetxt(f'./../Predictions/stats_{model_name}.txt', stats)
       
    return Erru, Erru2


