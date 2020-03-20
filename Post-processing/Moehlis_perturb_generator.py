#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:12:12 2020

@author: luca
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.io as sio
#import random
import os
from joblib import Parallel, delayed
import time

#%% Function definition

def rhs(t, a):
    """Compute the right hand side of Mohelis system of equations"""
#    a[0] = beta**2/Re - beta**2/Re*a[0]
#    a[1] = -(4*beta**2/3 + gamma**2)*a[1]/Re
    a_tp1 = np.ndarray((9,),float)
    a_tp1[0] = b2/Re - b2/Re*a[0] \
               - (3/2)**(1/2)*beta*gamma/k_abg*a[5]*a[7] \
               + (3/2)**(1/2)*beta*gamma/k_bg*a[1]*a[2]
    
    a_tp1[1] = - (4*b2/3 + g2)*a[1]/Re \
               + 5/3*(2/3)**(1/2)*g2/k_ag*a[3]*a[5] \
               - g2/(6)**(1/2)/k_ag*a[4]*a[6] \
               - alpha*beta*gamma/(6)**(1/2)/k_ag/k_abg*a[4]*a[7] \
               - (3/2)**(1/2)*beta*gamma/k_bg*a[0]*a[2] \
               - (3/2)**(1/2)*beta*gamma/k_bg*a[2]*a[8]
    
    a_tp1[2] = - (beta**2 + gamma**2)/Re*a[2] \
               + 2*alpha*beta*gamma/(6)**(1/2)/k_ag/k_bg*(a[3]*a[6]+ \
                                                           a[4]*a[5]) \
               + (b2*(3*a2+g2)-3*g2*(alpha**2 + gamma**2))/(6)**(1/2)/k_ag/k_bg/k_abg* \
                                                           a[3]*a[7] \
                                                           
    a_tp1[3] = - (3*a2 + 4*b2)/3/Re*a[3] - alpha/(6)**(1/2)*a[0]*a[4] \
               - 10/3/(6)**(1/2)*a2/k_ag*a[1]*a[5] \
               - (3/2)**(1/2)*alpha*beta*gamma/k_ag/k_bg*a[2]*a[6] \
               - (3/2)**(1/2)*a2*b2/k_ag/k_bg/k_abg*a[2]*a[7] \
               - alpha/(6)**(1/2)*a[4]*a[8]
               
    a_tp1[4] = - (a2 + b2)/Re*a[4] + alpha/(6)**(1/2)*a[0]*a[3] \
               + a2/(6)**(1/2)/k_ag*a[1]*a[6] \
               - alpha*beta*gamma/(6)**(1/2)/k_ag/k_abg*a[1]*a[7] \
               + alpha/(6)**(1/2)*a[3]*a[8] \
               + 2*alpha*beta*gamma/(6)**(1/2)/k_ag/k_bg*a[2]*a[5]

    a_tp1[5] = - (3*a2 + 4*b2 + 3*g2)/3/Re*a[5] + alpha/(6)**(1/2)*a[0]*a[6] \
               + (3/2)**(1/2)*beta*gamma/k_abg*a[0]*a[7] \
               + 10/3/(6)**(1/2)*(a2-g2)/k_ag*a[1]*a[3] \
               - 2*(2/3)**(1/2)*alpha*beta*gamma/k_ag/k_bg*a[2]*a[4] \
               + alpha/(6)**(1/2)*a[6]*a[8] \
               + (3/2)**(1/2)*beta*gamma/k_abg*a[7]*a[8]
               
    a_tp1[6] = - (a2 + b2 + g2)/Re*a[6] - alpha/(6)**(1/2)*(a[0]*a[5]+ \
                                                            a[5]*a[8]) \
               + (g2 - a2)/(6)**(1/2)/k_ag*a[1]*a[4] \
               + alpha*beta*gamma/(6)**(1/2)/k_ag/k_bg*a[2]*a[3] \
               
    a_tp1[7] = - (a2 + b2 + g2)/Re*a[7] \
               + 2*alpha*beta*gamma/(6)**(1/2)/k_ag/k_abg*a[1]*a[4] \
               + g2*(3*a2 - b2 + 3*g2)/(6)**(1/2)/k_ag/k_bg/k_abg*a[2]*a[3]                
               
    a_tp1[8] = - 9*b2/Re*a[8] + (3/2)**(1/2)*beta*gamma/k_bg*a[1]*a[2] \
               - (3/2)**(1/2)*beta*gamma/k_abg*a[5]*a[7]
    
    return a_tp1

## Eigenvector functions

def u_1(x,y,z):
    u1 = np.ndarray((3,),float)    
    u1[0] = math.sqrt(2)*math.sin(math.pi*y/2)
    u1[1] = 0
    u1[2] = 0
    return u1
    
def u_2(x,y,z):
    u2 = np.ndarray((3,),float)    
    u2[0] = 4/math.sqrt(3)*(math.cos(math.pi*y/2))**2*math.cos(gamma*z)
    u2[1] = 0
    u2[2] = 0
    return u2

def u_3(x,y,z):
    u3 = np.ndarray((3,),float)    
    u3[0] = 0
    u3[1] = 2*gamma*math.cos(math.pi*y/2)*math.cos(gamma*z)
    u3[2] = math.pi*math.sin(math.pi*y/2)*math.sin(gamma*z)
    u3 = u3*2/math.sqrt(4*g2+math.pi**2)
    return u3
    
def u_4(x,y,z):
    u4 = np.ndarray((3,),float)    
    u4[0] = 0 
    u4[1] = 0
    u4[2] = 4/math.sqrt(3)*(math.cos(math.pi*y/2))**2*math.cos(alpha*x)
    return u4

def u_5(x,y,z):
    u5 = np.ndarray((3,),float)    
    u5[0] = 0 
    u5[1] = 0
    u5[2] = 2*math.sin(math.pi*y/2)*math.sin(alpha*x)
    return u5

def u_6(x,y,z):
    u6 = np.ndarray((3,),float)    
    u6[0] = - gamma*(math.cos(math.pi*y/2))**2*math.cos(alpha*x)*math.sin(gamma*z)
    u6[1] = 0
    u6[2] = alpha*math.sin(alpha*x)*(math.cos(math.pi*y/2))**2*math.cos(gamma*z)
    u6 = u6*4*math.sqrt(2)/math.sqrt(3*(a2 + g2))
    return u6

def u_7(x,y,z):
    u7 = np.ndarray((3,),float)    
    u7[0] = gamma*math.sin(math.pi*y/2)*math.sin(alpha*x)*math.sin(gamma*z)
    u7[1] = 0
    u7[2] = alpha*math.sin(math.pi*y/2)*math.cos(alpha*x)*math.cos(gamma*z)
    u7 = u7*2*math.sqrt(2)/math.sqrt(a2 + g2)
    return u7

def u_8(x,y,z):
    u8 = np.ndarray((3,),float)    
    u8[0] = math.pi*alpha*math.sin(math.pi*y/2)*math.sin(alpha*x)*math.sin(gamma*z)
    u8[1] = 2*(a2 + g2)*math.cos(math.pi*y/2)*math.cos(alpha*x)*math.sin(gamma*z)
    u8[2] = - gamma*math.pi*math.sin(math.pi*y/2)*math.cos(alpha*x)*math.cos(gamma*z)
    u8 = u8*2*math.sqrt(2)/math.sqrt((a2 + g2)*(4*a2 + 4*g2 + math.pi**2))
    return u8

def u_9(x,y,z):
    u9 = np.ndarray((3,),float)    
    u9[0] = math.sqrt(2)*math.sin(3*math.pi*y/2)
    u9[1] = 0
    u9[2] = 0
    return u9

def eig(n,x,y,z):
    switcher={
            1:u_1,
            2:u_2,
            3:u_3,
            4:u_4,
            5:u_5,
            6:u_6,
            7:u_7,
            8:u_8,
            9:u_9
            }
    func=switcher.get(n,lambda :'Invalid')
    return func(x,y,z)

#%% Configuration
    
cur_path = os.path.dirname(__file__)
ds_ode_path = '/../../Datasets/'

plt.close('all')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

## Parameters of the simulation

Re = 400.0              # Reynolds number
L_x = 4*math.pi         # Channel streamwise dimension
L_z = 2*math.pi         # Channel spanwise dimension
y_min = -1.0            # Channel height
y_max = 1.0
t0 = 0.0                # Initial integration time
T = 4050.0              # Final integration time
step_size = 0.1        # TO BE SET SO THAT T_save IS INT
T_save = int(4000/step_size)
Start_num = 0
Num_fields = 1
## Derived parameters

alpha = 2*math.pi/L_x
beta = math.pi/2 
gamma = 2*math.pi/L_z

a2 = alpha**2
b2 = beta**2
g2 = gamma**2
   
k_ag = math.sqrt(alpha**2 + gamma**2)
k_bg = math.sqrt(beta**2 + gamma**2)
k_abg = math.sqrt(alpha**2 + beta**2 + gamma**2) 

## Computation starts here
t_perturb = np.array([500])#, 1000, 1500, 2000])
delta_A0 = 1e-6

seq_path = cur_path+ds_ode_path
if not os.path.exists(seq_path):
    os.mkdir(seq_path)

seq_path = seq_path+f'perturbed_{delta_A0}/'
if not os.path.exists(seq_path):
    os.mkdir(seq_path)

t = np.arange(t0, T, step_size)
 
a0 = np.array([1, 0.07076, -0.07066, 0.04, 0, 0, 0, 0, 0])

n_fields = 0

turbulent = False
wrt = {} 
n_attempts = 0
i = 0

while n_fields < Num_fields: 
    if not turbulent:    
        perturb = np.random.uniform(-1,1,(9,))            
    
        Del_A0_bar = np.linalg.norm(perturb, axis=0)             
        perturb *= delta_A0/Del_A0_bar
    
    n_attempts += 1
    
#    for i in range(0,len(t_perturb)):        
    sol_pre = integrate.solve_ivp(rhs, (t0, t_perturb[i]+1), a0, max_step=0.1)
    sol_perturb = np.copy(sol_pre.y[:,-2])
    #print(sol_pre.y[:,-2])
    
    #print('Delta A0 pre = ' + str(Del_A0_bar))
    sol_perturb += perturb
    print(perturb)
#        Del_A0_check = np.linalg.norm(sol_pre.y[:,-2]-sol_perturb)
    print(sol_pre.y[:,-2]-sol_perturb)
    print('Delta A0 = ' + str(np.linalg.norm(sol_pre.y[:,-2]-sol_perturb)))
    
    t_post = np.arange(np.ceil(sol_pre.t[-2]), T, 1.0)
    sol_post = integrate.solve_ivp(rhs, (sol_pre.t[-2], T), sol_perturb, \
                                  max_step=0.1, t_eval=t_post)
    #sol_unpert = integrate.solve_ivp(rhs, (sol_pre.t[-2], T), sol_perturb, max_step=0.1, t_eval=t_post)
    sol = integrate.solve_ivp(rhs, (t0, T), a0, max_step=0.1, t_eval=t)
     
    sol_wrt = np.concatenate((sol.y[:,:int(np.ceil(sol_pre.t[-2]))], sol_post.y), axis=1)

    wrt['testSeq'] = sol.y[:,:T_save]
    wrt['pertSeq'] = sol_wrt[:,:T_save]
#    
    if (sol.y[0,np.where(sol.t == next(x for x in sol.t if x > 4000))] < 0.98 and \
        sol_post.y[0,np.where(sol_post.t == next(x for x in sol_post.t if x > 4000))] < 0.98):
        print(str(i)+' sequences generated with pert at t='+str(t_perturb[i]))
        save_name=seq_path+'pert_'+str(n_fields+1)+'_'+str(t_perturb[i])
        sio.savemat(save_name, wrt, appendmat=False)
        turbulent=True
        if i == len(t_perturb)-1:
            for ii in range(0, len(t_perturb)):
                re_name=seq_path+'pert_'+str(n_fields+1)+'_'+str(t_perturb[ii])
                os.rename(re_name,re_name+'.mat')
            turbulent=False
            n_fields += 1
    else:
        for ii in range(0, i):
                re_name=seq_path+'pert_'+str(n_fields+1)+'_'+str(t_perturb[ii])
                os.remove(re_name)
        turbulent=False
        break








