
"""

272_Project 2

Lennard-Jones Potential Simulation 
using Metropolis Monte-Carlo Method

-Group 2-
David Houshangi
Yash Maheshwaran
Yejin Yang
Christian Fernandez
Seungho Yoo


"""

import numpy as np
import matplotlib.pyplot as plt

#1. Initialize current locations of N particles within a square box of side length 2R.
#   Particles are placed with uniform random distribution in both x and y directions.
def PlotLocation(R = 60, N = 100):
    Xinit = np.random.uniform(-R, R, (N,))
    Yinit = np.random.uniform(-R, R, (N,))
    return Xinit, Yinit


#2. Define the pairwise Lennard–Jones potential between particles.
#   a: strength of repulsive term (r^-12)
#   b: strength of attractive term (r^-6)
#   Division by zero is avoided by setting diagonal distances to infinity.
def Potential(Dx, Dy, a = 1, b = 1): 
    r = np.sqrt(Dx**2 + Dy**2)                  # Pairwise distances
    np.fill_diagonal(r, np.inf)                 # Avoid self-interaction
    Phi = a/r**12 - b/r**6                      # Lennard-Jones potential
    np.fill_diagonal(Phi,0)                     # Zero out self-interaction
    return Phi


#3. Compute total potential energy for each particle given positions.
#   Uses periodic wrapping to account for periodic boundary conditions.
def DisToPotential(Xinit, Yinit, a, b, Ones):
    # Create tiled arrays to compute pairwise differences
    N = len(Xinit)
    Xi_tile = np.tile(Xinit, (N,1)) 
    Yi_tile  = np.tile(Yinit, (N,1))
    
    Dx = Xi_tile - Xi_tile.T
    Dy = Yi_tile - Yi_tile.T
    
    Phi = Potential(Dx, Dy, a, b)
    Utot = np.dot(Phi, Ones)                    # Total potential per particle (sum over j)
    return Utot


# 4. Main Metropolis Monte Carlo loop: propose and accept/reject particle moves.
#   - All particles are moved simultaneously.
#   - Propose simultaneous Gaussian displacements, dx, dy ~ N(0, sigma^2) (sigma sets step size)
#   - Metropolis criterion: accept move if dU < 0 or with probability exp(-dU/T).
def MoveParticle(Xinit, Yinit, Niter = 1000, a = 1, b = 1, T = 10, sigma = 0.01, R=60):
    N = len(Xinit)
    Ones = np.ones((N,))

    for n in range(Niter):
        # Generate trial moves for all particles
        dx = np.random.normal(0, sigma, (N,)) #delta is to adjust step size
        dy = np.random.normal(0, sigma, (N,))
        
        Xtri = Xinit + dx
        Ytri = Yinit + dy
        
        # Compute energy change
        Uold = DisToPotential(Xinit, Yinit, a, b, Ones) 
        Utri = DisToPotential(Xtri, Ytri, a, b, Ones) 
        dU = Utri - Uold
        
        # Metropolis acceptance step
        rho = np.random.uniform(0,1, size = dU.shape) 
        criteria = np.clip(-dU / T, 0, 700 )  # Avoid overflow
        accept = (dU < 0) | (rho < np.exp(criteria))
        move_indices = np.flatnonzero(accept)
        
        # Update accepted particles
        for i in move_indices:
            Xinit[i] = Xtri[i]
            Yinit[i] = Ytri[i]
            
        # Plot positions periodically for visualization
        if not n%100:
            plt.scatter(Xinit, Yinit, c='gray', alpha=0.5, s=30)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'after {n} iterations\na = {a}, b = {b}, T = {T}, N = {N}')
            plt.show()


#5. Simulation wrapper class for easier parameter control.
class SimulateParticles:
    def __init__(self, N=100, R=15, a=1, b=1, T=10, sigma=0.01):
        self.N = N
        self.R = R
        self.a = a
        self.b = b
        self.T = T
        self.sigma = sigma
        self.Xinit, self.Yinit = PlotLocation(R, N)

    def run(self, Niter=1000):
        MoveParticle(self.Xinit, self.Yinit, Niter, 
                     self.a, self.b, self.T, self.sigma, self.R, )


# Example run:
sim = SimulateParticles(N=200,a = 4, b = 4, T = 1, sigma = 0.01)
sim.run(Niter=10_000)
