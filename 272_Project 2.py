"""

272_Project 2
Yejin Yang
Refer to 272_03 Problem set

"""

import numpy as np
import matplotlib.pyplot as plt

#1. current locations
def PlotLocation(R = 50, N = 100):
    Xinit = np.random.uniform(-R, R, (N,1))
    Yinit = np.random.uniform(-R, R, (N,1))
    return Xinit, Yinit

#2.define potential
def Potential(Dx, Dy, a = 1, b = 1, N = 100): #N repeatedly input?
    r = np.sqrt(Dx**2 + Dy**2)
    
    #to avoid error from division by zero
    eps = 1e-10
    Eye = np.eye(N)
    r = r + Eye * (1/eps)
    
    Phi = a/r**12 - b/r**6
    np.fill_diagonal(Phi,0)
    return Phi


#3.call Potential, calculates the distances and the total potential
def DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones):
    Xi_tile = np.tile(Xinit, (1,N)) #Xinit from PlotLocation
    Yi_tile  = np.tile(Yinit, (1,N))
    
    Dx = Xi_tile - Xi_tile.T
    Dy = Yi_tile - Yi_tile.T
    
    Phi = Potential(Dx, Dy, a, b, N)
    Utot = np.dot(Phi, Ones)
    return Utot


#4. calculate if a particle is allowed to move
def MoveParticle(Xinit, Yinit, N = 100, Niter = 1000, a = 1, b = 1, T = 1, delta = 1):
    
    N = len(Xinit)
    Eye = np.eye(N)
    Ones = np.ones((N,))
    
    Utot = DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones) #what input?
    #then it generates Xi_tile, Yi_tile, Dx, Dy, Dxy, Phi, Utot?

    for n in range(Niter):
        i = np.random.randint(N)  #pick one particle
        dx = delta * np.random.choice([-1, 1])
        dy = delta * np.random.choice([-1, 1])
    
        Xtri = np.copy(Xinit)
        Ytri = np.copy(Yinit)
        
        Xtri[i] += dx
        Ytri[i] += dy
    
        U_old = DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones)[i]
        U_new = DisToPotential(Xtri, Ytri, a, b, N, Eye, Ones)[i]
        dU = U_new - U_old
    
        rho = np.random.rand()
        if dU < 0 or rho < np.exp(-dU / T):
            Xinit[i] = Xtri[i]
            Yinit[i] = Ytri[i]
            Utot[i] = U_new
            
        #Plotting
        if not n%100:
            plt.scatter(Xinit, Yinit, c='gray', alpha=0.5)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'after {n} iterations\nT = {T}, a = {a}, b = {b}')
            plt.show()

#MoveParticle()

#5. runs the simulate
class SimulateParticles:
    def __init__(self, N = 100, R = 50):
        self.N = N
        self.R = R
        self.Xinit, self.Yinit = PlotLocation(R, N)

    def run(self, Niter=1000, a=1, b=1, T=1, delta=1):
        MoveParticle(self.Xinit, self.Yinit, self.N, Niter, a, b, T, delta)
        
        
sim = SimulateParticles(N=200)
sim.run(Niter=6000, T = 1)
#sim.run(Niter=1000, T = 10)
