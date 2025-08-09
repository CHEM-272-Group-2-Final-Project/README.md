"""

272_Project 2
Yejin Yang
Refer to 272_03 Problem set
MOVE ALL PARTICLES AT A TIME

"""

import numpy as np
import matplotlib.pyplot as plt

#1. currnet locations
def PlotLocation(R = 50, N = 100):
    Xinit = np.random.uniform(-R, R, (N,))
    Yinit = np.random.uniform(-R, R, (N,))
    return Xinit, Yinit

#2.define potential
def Potential(Dx, Dy, a = 1, b = 1, N = 100): 
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
def MoveParticle(Xinit, Yinit, N = 100, Niter = 1000, a = 1, b = 1, T = 10, delta = 0.1):
    
    N = len(Xinit)
    Eye = np.eye(N)
    Ones = np.ones((N,))
    
    Utot = DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones)
    #then it generates Utot?

    #check step size adjustment constant "delta"
    #r0 = (a / b)**(1/6) 
    #delta = 0.1 #when r closes to r0*0.1, aggregatioin starts?
    #print("compare 0.1*r0 to delta: ", 0.1*r0, delta)
    
    for n in range(Niter):
        dx = delta * np.random.choice([-1, 1],(N,1)) #delta is to adjust step size
        dy = delta * np.random.choice([-1, 1],(N,1))
        #print("dx, dy", dx, dy)
        Xtri = np.copy(Xinit)
        Ytri = np.copy(Yinit)
        
        Xtri += dx
        Ytri += dy
        #print("Xtri Ytri: ", Xtri, Ytri)
        
        Uold = DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones) #old
        Utri = DisToPotential(Xtri, Ytri, a, b, N, Eye, Ones) #new
        dU = Utri - Uold
        
        rho = np.random.uniform(0,1) #rho should be differnet for each particle for strictly speaking, for detailed balance
        accept = (dU < 0) | (rho < np.exp(-dU / T))
        move_indices = np.flatnonzero(accept)
        
        for i in move_indices:
            Xinit[i] = Xtri[i]
            Yinit[i] = Ytri[i]
            Utot[i] = Utri[i]
            
        #Plotting
        if not n%500:
            plt.scatter(Xinit, Yinit, c='gray', alpha=0.5, s=30)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'after {n} iterations\na = {a}, b = {b}, T = {T}, N = {N}')
            #plt.savefig(f'after {n} iterations, a = {a}, b = {b}, T = {T}, N = {N}.png')
            plt.show()

#5. runs the simulate
class SimulateParticles:
    def __init__(self, N=100, R=50, a=1, b=1, T=10, delta=1):
        self.N = N
        self.R = R
        self.a = a
        self.b = b
        self.T = T
        self.delta = delta
        self.Xinit, self.Yinit = PlotLocation(R, N)

    def run(self, Niter=1000):
        MoveParticle(self.Xinit, self.Yinit, self.N, Niter, self.a, self.b, self.T, self.delta)
        
        
sim = SimulateParticles(N=500,a = 1, b = 1)
#sim = SimulateParticles(N=500,a = 1, b = 20)#greater attraction
#sim = SimulateParticles(N=500,a = 20, b = 1)#greater repulsion X

#sim = SimulateParticles(N=500,T = 1000)#higher temperature X
#sim = SimulateParticles(N=500,T = 1) #lower temperature X

#sim = SimulateParticles(N=50)#less population
#sim = SimulateParticles(N=2000)#more population
sim.run(Niter=1001)

