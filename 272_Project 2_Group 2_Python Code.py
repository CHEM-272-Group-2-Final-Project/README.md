
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
def PlotLocation(R = 15, N = 100):
    Xinit = np.random.uniform(-R, R, (N,))
    Yinit = np.random.uniform(-R, R, (N,))
    return Xinit, Yinit


#2. Define the pairwise Lennard–Jones potential between particles.
#   a: strength of repulsive term (r^-12)
#   b: strength of attractive term (r^-6)
#   Division by zero is avoided by setting diagonal distances to infinity.
def Potential(Dx, Dy, a = 1, b = 1, N = 100): 
    r = np.sqrt(Dx**2 + Dy**2)                  # Pairwise distances
    np.fill_diagonal(r, np.inf)                 # Avoid self-interaction
    Phi = a/r**12 - b/r**6                      # Lennard-Jones potential
    np.fill_diagonal(Phi,0)                     # Zero out self-interaction
    return Phi


#3. Compute total potential energy for each particle given positions.
#   Uses periodic wrapping to account for periodic boundary conditions.
def DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones, L):
    # Create tiled arrays to compute pairwise differences
    Xi_tile = np.tile(Xinit, (N,1)) 
    Yi_tile  = np.tile(Yinit, (N,1))
    
    Dx = Xi_tile - Xi_tile.T
    Dy = Yi_tile - Yi_tile.T
    
    # Apply periodic wrapping 
    Dx -= L * np.round(Dx / L)
    Dy -= L * np.round(Dy / L)
    
    Phi = Potential(Dx, Dy, a, b, N)
    Utot = np.dot(Phi, Ones)                    # Total potential per particle (sum over j)
    return Utot


# 4. Main Metropolis Monte Carlo loop: propose and accept/reject particle moves.
#   - All particles are moved simultaneously by ±delta in x and y.
#   - Periodic boundary conditions applied after the move.
#   - Metropolis criterion: accept move if dU < 0 or with probability exp(-dU/T).
def MoveParticle(Xinit, Yinit, N = 100, Niter = 1000, a = 1, b = 1, T = 10, delta = 1, R=50):
    L = R*2                                    # Box length for periodic boundaries
    N = len(Xinit)
    Eye = np.eye(N)
    Ones = np.ones((N,))

    for n in range(Niter):
        # Generate trial moves for all particles
        dx = delta * np.random.choice([-1, 1],(N,)) #delta is to adjust step size
        dy = delta * np.random.choice([-1, 1],(N,))
        
        Xtri = np.copy(Xinit)
        Ytri = np.copy(Yinit)
        
        Xtri += dx
        Ytri += dy
        
        # Apply periodic boundary conditions (wrap into [-R, R])
        Xtri = (Xtri + R) % L - R
        Ytri = (Ytri + R) % L - R
        
        # Compute energy change
        Uold = DisToPotential(Xinit, Yinit, a, b, N, Eye, Ones, L) 
        Utri = DisToPotential(Xtri, Ytri, a, b, N, Eye, Ones, L) 
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
    def __init__(self, N=100, R=15, a=1, b=1, T=10, delta=1):
        self.N = N
        self.R = R
        self.a = a
        self.b = b
        self.T = T
        self.delta = delta
        self.Xinit, self.Yinit = PlotLocation(R, N)

    def run(self, Niter=1000):
        MoveParticle(self.Xinit, self.Yinit, self.N, Niter, 
                     self.a, self.b, self.T, self.delta, self.R, )


# Example run:
sim = SimulateParticles(N=500,a = 1, b = 1, T = 10, delta = 0.01)
sim.run(Niter=10_000)