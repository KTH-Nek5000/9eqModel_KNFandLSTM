"""
HDMDs.py
---------------------
Python class for Hankel dynamic mode decomposition (HDMD).

Parameters:
    train: 2-D array (n_samples, n_features)
            Time series data for training of the KNF model. Each column of 
            train represents a variable, and each row a single observation of 
            all those variables.
            
    q: int
            The delay-embedding dimension
            
    r_opt: bool, default=False
            If r_opt is True, the model uses optimal hard threshold for 
            svd rank truncation. Otherwise, the rank of truncation should be given
            based on the energy contribution. Code is adapted from:
            Gavish, Matan, and David L. Donoho. 
            "The optimal hard threshold for singular values is 4/sqrt(3)" 
            IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.    
            http://arxiv.org/abs/1305.5870
            
    er: float
            Rank truncation for operator A. The ranks with less energy than er 
            will be truncated.
            

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
from scipy import linalg as la
from optht import optht
from matplotlib import pyplot as plt

class HDMD():
    def __init__(self, train, q, r_opt = False, er = 1e-7):
        self.train = train
        self.q = q
        self.r_opt = r_opt
        self.er = er
       
    def fit(self):

        self.n = self.train.shape[1]
        self.A, self.phi, self.lam, self.U_hdmd = self.HDMD()
        
        return self

    
    def Hankel(self, data, q):
        nt = data.shape[0]
        x = [data[i-q:i] for i in range(q, nt+1)]
        x = np.array(x); x = x.reshape((x.shape[0], -1))
        x = x.T
        return x[:, :-1], x[:, 1:]

    def Delay(self, data, q):
        x = [data[i-q:i] for i in range(q, q+1)]
        x = np.array(x); x = x.reshape((x.shape[0], -1))
        x = x.T
        return x
    
      
    def HDMD(self):
        x = self.train
        q = self.q
        xH, yH = self.Hankel(x, q)
        
        U, S, Vh = la.svd(xH, full_matrices=False)
        if self.r_opt:
            r = optht(xH, S)
        else:
            r_p = np.where(S > self.er)
            r = np.size(r_p)

        V = Vh.T
        U = U[:, :r];
        S = S[:r];
        V = V[:, :r];
        
        S = np.diag(S)
    
        A = U.T @ yH @ V @ la.inv(S)
        
        lam, W = la.eig(A)
        
        phi = yH @ V @ la.inv(S) @ W;
        
        return A, phi, lam, U
    
    def predict(self, x0, npred):
        q = self.q
        n = self.n
        x0H = self.Delay(x0, q)
        reg_xr = la.lstsq(self.U_hdmd, x0H)
        x0_r = reg_xr[0]
           
        pred = []
        for i in range(npred):
        
            x_new = self.A @ x0_r
            x0_r = x_new.copy()
            X_new = self.U_hdmd @ x_new
                       
            pred.append(X_new[:n])

        pred = np.array(pred).squeeze()

        return pred

    def pltLambda(self):
        lam = self.lam
        plt.figure(figsize = (4, 4))
        plt.plot(np.real(lam), np.imag(lam), 'o')
        plt.plot(np.cos(np.arange(0, 2 * np.pi, np.pi/180)), np.sin(np.arange(0, 2 * np.pi, np.pi/180)), '--k')
        plt.xlabel('Re($\lambda$)')
        plt.ylabel('Im($\lambda$)')
        plt.grid()
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])