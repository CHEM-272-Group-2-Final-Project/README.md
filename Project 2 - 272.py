# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 15:27:58 2025

@author: Christian
"""

import numpy as np
import matplotlib.pyplot as plt

def CoordsToPotential(Xin, Yin, a, b, N):
    
    X      = np.tile(Xin, (N,1))
    Y      = np.tile(Yin, (N,1))
    
    Dx     = X - X.transpose()
    Dy     = Y - Y.transpose()
    r = np.sqrt(((Dx)**2)+((Dy)**2))    #distance between particles
    
    LJ = lambda r: (a/(r**12)) - (b/(r**6)) 
    Phi = LJ(r)
    np.fill_diagonal(Phi, 0)

    Utot = np.dot(Phi, np.ones(N))
    
    return Utot

def Utot_Move_Particle(Niter: int = 1001, N: int = 1000, a: float = 1, b: float = 1, delta = 0.01, T = 25):
    
    Xinit = np.random.uniform(0,10,(N,))
    Yinit = np.random.uniform(0,10,(N,))
    
    Utot  = CoordsToPotential(Xinit, Yinit, a, b, N)
    
    for n in range(Niter):
        
        dx = delta* np.random.choice([-1, 1],(N,))
        dy = delta* np.random.choice([-1, 1],(N,))

        X = Xinit + dx
        Y = Yinit + dy

        Utot_new = CoordsToPotential(X, Y, a, b, N)

        accepted_moves = np.argwhere(Utot_new < Utot) #creats a 2D (N,1) array of indices for accepted moves
        accepted_moves = np.reshape(accepted_moves, -1) #resize into 1D (N, ) array for indexing purposes later

        unfavorable_moves = np.argwhere(Utot_new > Utot)
        unfavorable_moves = np.reshape(unfavorable_moves, -1)
        
         #size of this?
        frac = np.exp(-(Utot_new[unfavorable_moves] - Utot[unfavorable_moves])/ T)
        rho = np.random.uniform(0,1,len(frac))
        favorable_moves = np.argwhere(rho < frac)
        favorable_moves = np.reshape(favorable_moves, -1)
        
        accepted_unfavorable_moves = unfavorable_moves[favorable_moves]
        
        Xinit[accepted_unfavorable_moves] = X[accepted_unfavorable_moves]
        Yinit[accepted_unfavorable_moves] = Y[accepted_unfavorable_moves]
        Utot[accepted_unfavorable_moves] = Utot_new[accepted_unfavorable_moves]
        

        Xinit[accepted_moves] = X[accepted_moves]
        Yinit[accepted_moves] = Y[accepted_moves]
        Utot[accepted_moves] = Utot_new[accepted_moves]


        if not n%100:
            
            plt.scatter(Xinit, Yinit, s= 10, marker = "o", color = 'grey')
            plt.title('after ' + str(n) + ' iterations')
            plt.show()
            
Utot_Move_Particle()   