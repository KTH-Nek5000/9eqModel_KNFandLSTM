"""
KNFs.py
---------------------
Python class for Koopman-based framework with nonlinear forcing (KNF).

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
            
    ep: float
            Rank truncation for operator B. The ranks with less energy than ep 
            will be truncated.
            
    er: float
            Rank truncation for operator A. The ranks with less energy than er 
            will be truncated.
            
    polyorder: int
            The maximum order of the polynomial nonlinear functions for 
            construction of the forcing term.
            
    sin_cos: bool or int, default=False
            The maximum order of the trigonometric nonlinear functions for 
            construction of the forcing term. If False, trigonometric nonlinear 
            functions would not be considered.
            
    cons: bool
            If True, a constant would be considered in forcing term.
    
    excludes: list
            A list of orders of polynomial nonlinear functions that should be 
            excluded from the forcing term.
            
    sparse: bool, default=False
            If True, sparsity promotion would be applied. 
            
    threshold: float
            The threshold for sparsity promotion. Coefficients with a value less
            than the threshold will be replaced by zero.
            
    max_itr: int
            Maximum iteration for sparsity promotion.
            
    limit: float
            If the maximum value of the predictions is larger than the limit, 
            the prediction iterations would be stopped. 
            

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
from sklearn.linear_model import ridge_regression, LinearRegression
from itertools import compress



class KNF():
    def __init__(self, train, q, r_opt = False, ep = 1e-3, er = 1e-3, 
                 polyorder = 2, sin_cos = False, cons = True, excludes = [],
                 sparse = False, threshold = 1e-3, max_itr = 20, limit = 0.999):
        self.train = train
        self.q = q
        self.r_opt = r_opt
        self.ep = ep
        self.er = er
        self.o = polyorder
        self.sin_cos = sin_cos
        self.cons = cons
        self.excludes = excludes
        self.n_d = train.shape[1] 
        self.n_h = self.n_d * self.q 
        self.sparse = sparse
        self.threshold = threshold
        self.max_itr = max_itr
        self.limit = limit
       
    def fit(self):

        self.n = self.train.shape[1]
        self.A, self.B, self.phi, self.lam, self.U_dmdc, self.U1, self.U2 = self.HDMDc()
        
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
    
    def nonlinear_dict(self):
        flist = []
        nlist = []
        r = self.n_d
        if self.cons:
            flist.append('1')
            nlist.append([1])
        # Poly order 2
        if self.o >= 2 and 2 not in self.excludes:
            for i in range(r):
                for j in range(i, r):
                    flist.append(f'a_{i}*a_{j}')
                    nlist.append([i, j])
        # Poly order 3
        if self.o >= 3 and 3 not in self.excludes:
            for i in range(r):
                for j in range(i, r):
                    for k in range(j, r):
                        flist.append(f'a_{i}*a_{j}*a_{k}')
                        nlist.append([i, j, k])

        # Poly order 4
        if self.o >= 4 and 4 not in self.excludes:
            for i in range(r):
                for j in range(i, r):
                    for k in range(j, r):
                        for l in range(k, r):
                            flist.append(f'a_{i}*a_{j}*a_{k}*a_{l}')
                            nlist.append([i, j, k, l])
        # Sin and Cos
        if self.sin_cos != 0:
            for k in range(1, self.sin_cos + 1):
                for i in range(r):
                    flist.append(f'sin({k}*a{i})')
                    flist.append(f'cos({k}*a{i})')
                    nlist.append(['sin', k, i]) 
                    nlist.append(['cos', k, i])
        self.nonlinear_funcs = flist
        self.nonlinear_inds = nlist
        
        return self
    
    def nonlinear_obs(self, data):
        
        nt = data.shape[0]
        f = np.empty((nt, 0))
        
        inds = list(compress(self.nonlinear_inds, self.ind_n))
        
        if self.cons:
            index = [i for i in inds if len(i) == 1]
            for m in index:
                fi = np.ones((nt, 1))
                f = np.concatenate((f, fi), axis = 1)
        # Poly order 2
        if self.o >= 2 and 2 not in self.excludes:
            index = [i for i in inds if len(i) == 2]
            for m in index:
                i, j = m
                fi = data[:, i] * data[:, j]
                fi = fi.reshape((-1, 1))
                f = np.concatenate((f, fi), axis = 1)
        # Poly order 3
        if self.o >= 3 and 3 not in self.excludes:
            index = [i for i in inds if len(i) == 3 and type(i[0]) == int]
            for m in index:
                i, j, k = m
                fi = data[:, i] * data[:, j] * data[:, k]
                fi = fi.reshape((-1, 1))
                f = np.concatenate((f, fi), axis = 1)
        # Poly order 4
        if self.o >= 4 and 4 not in self.excludes:
            index = [i for i in inds if len(i) == 4]
            for m in index:
                i, j, k, l = m
                fi = data[:, i] * data[:, j] * data[:, k] * data[:, l]
                fi = fi.reshape((-1, 1))
                f = np.concatenate((f, fi), axis = 1)
        # Sin and Cos
        if self.sin_cos != 0:
            index = [i for i in inds if i[0] == 'sin' or i[0] == 'cos']
            for m in index:
                t, k, i = m
                if t == 'sin':
                    fi_sin = np.sin(k * data[:, i]).reshape((-1, 1))
                    f = np.concatenate((f, fi_sin), axis = 1)
                if t == 'cos':
                    fi_cos = np.cos(k * data[:, i]).reshape((-1, 1))
                    f = np.concatenate((f, fi_cos), axis = 1)
           
        return f
    
    
    def regression(self, xH, yH, fH):
        n_h = xH.shape[0]
        
        U_tau, S_tau, Vh_tau = la.svd(yH, full_matrices=False)
        if self.r_opt:
            r = optht(yH, S_tau)
        else:
            r_p = np.where(S_tau > self.er)
            r = np.size(r_p)
        U_tau = U_tau[:, :r]
        S_tau = np.diag(S_tau[:r])
        
        c = np.concatenate((xH, fH), axis = 0)
        
        U, S, Vh = la.svd(c, full_matrices=False)
        if self.r_opt:
            p = int(self.alpha * r) #or optht(c, S)
        else:
            s_p = np.where(S > self.ep)
            p = np.size(s_p)
        V = Vh.T
        U = U[:, :p];
        S = S[:p];
        V = V[:, :p];
        
        S = np.diag(S)
        
        U1 = U[:n_h]
        U2 = U[n_h:]
        
        A = U_tau.T @ yH @ V @ la.inv(S) @ U1.T @ U_tau
        B = U_tau.T @ yH @ V @ la.inv(S) @ U2.T
        
        return A, B, V, S, U1, U2, U_tau
        
    def ridg_reg(self, y, x, fx, ind):
        X = np.concatenate((x, fx), axis = 0)[ind].T
        coef = ridge_regression(X, y, alpha = 0.05)

        return coef
    
    def sparse_coeffs(self, coef, ind, dim):
        """Perform thresholding of the weight vector(s)
        """
        c = np.zeros(dim)
        c[ind] = coef
        ind = np.abs(c) >= self.threshold
        c[~ind] = 0
        return c, ind
    
    def sparsity_cal(self):
        x = self.train       
        n_linf = self.n_d
        n_nonlinf = len(self.nonlinear_funcs)
        
        coef = np.zeros((n_linf, n_linf + n_nonlinf))
        self.coef = coef.copy()
        ind = np.ones(coef.shape, dtype = bool)
        self.ind = ind.copy()
        dim = n_linf + n_nonlinf
        
        self.ind_n = np.ones((n_nonlinf,), dtype = bool)
        
        xH, yH = self.Hankel(x, 1)
        fn = self.nonlinear_obs(x)      
        fH, _ = self.Hankel(fn, 1)
        
        
        if self.sparse:
        
            for _ in range(self.max_itr):
                for i in range(self.n_d):
                    yHi = yH[i]
                    ind_i = ind[i]
                    coef_i = self.ridg_reg(yHi, xH, fH, ind_i)
                    coef_i, ind_i = self.sparse_coeffs(coef_i, ind_i, dim)
                    
                    coef[i] = coef_i
                    ind[i] = ind_i
                if np.sum(self.coef - coef) == 0: # No change
                    break
                self.coef = coef.copy()
                self.ind = ind.copy()
            
            x = np.concatenate((xH, fH), axis = 0)
            for i in range(self.n_d):
                xi = x[ind[i]].T
                coef_i = LinearRegression(fit_intercept=False,
                                          normalize=False).fit(xi, yH[i]).coef_
                coef_i, ind_i = self.sparse_coeffs(coef_i, ind[i], dim)
                coef[i] = coef_i
                ind[i] = ind_i
                
            self.coef = coef.copy()
            self.ind = ind.copy()
            
            indb = ind[:, n_linf:]
            ind_n = np.sum(indb, axis = 0) != 0
            self.ind_n = ind_n
            
        return self
    
    def HDMDc(self):
        x = self.train
        q = self.q
        self.nonlinear_dict()
        self.sparsity_cal()
        
        xH, yH = self.Hankel(x, q)
        fn = self.nonlinear_obs(x)
        fH, _ = self.Hankel(fn, q)
        self.alpha = fH.shape[0]/xH.shape[0]

        A, B, V, S, U1, U2, U_tau = self.regression(xH, yH, fH)
        
        lam, W = la.eig(A)
        
        phi = yH @ V @ la.inv(S) @ U1.T @ U_tau @ W;
               
        return A, B, phi, lam, U_tau, U1, U2
    
    def predict(self, x0, npred):
        
        q = self.q
        n = self.n
        conv = True
        x0H = self.Delay(x0, q)
        reg_xr = la.lstsq(self.U_dmdc, x0H)
        x0_r = reg_xr[0]
    
        f0 = self.nonlinear_obs(x0)
        f0H = self.Delay(f0, q)
        
        pred = []
        for i in range(npred):
        
            x_new = self.A @ x0_r + self.B @ f0H
            x0_r = x_new.copy()
            X_new = self.U_dmdc @ x_new
            
            if np.max(X_new) > self.limit:
                conv = False
                break
                      
            pred.append(X_new[:n])
                    
            X_new = X_new.T
            X_new = X_new.reshape((q, n))
            
            f0 = self.nonlinear_obs(X_new)
            f0H = self.Delay(f0, q)
        pred = np.array(pred)[:, :, 0]
        if conv:
            return pred
        else:
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