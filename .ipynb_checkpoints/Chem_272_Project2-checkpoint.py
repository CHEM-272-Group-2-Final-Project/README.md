"""

272_Project 2
Seungho Yoo

"""


import numpy as np
import matplotlib.pyplot as plt

# 1. Generate initial locations
def PlotLocations(R=50, N=100):
    Xinit = np.random.uniform(-R, R, N)
    Yinit = np.random.uniform(-R, R, N)
    return Xinit, Yinit

# 2. Lennard-Jones potential matrix
def Potential(Dx, Dy, a=1, b=1):
    N = Dx.shape[0]
    r2 = Dx**2 + Dy**2
    Eye = np.eye(N)
    r2 += Eye  # prevent divide by zero in-place
    Phi = (a / r2**6) - (b / r2**3)
    np.fill_diagonal(Phi, 0)  # ensure no self-interaction
    return Phi

# 3. Convert positions to total potential
def DisToPotential(X, Y, a=1, b=1):
    N = len(X)
    X_tile = np.tile(X, (N, 1))
    Y_tile = np.tile(Y, (N, 1))
    Dx = X_tile - X_tile.T
    Dy = Y_tile - Y_tile.T

    Phi = Potential(Dx, Dy, a, b)
    Utot = np.dot(Phi, np.ones(N))
    return Utot

# 4. Metropolis move step
def MoveParticle(Xinit, Yinit, N=100, Niter=1000, a=1, b=1, T=1, delta=1):
    N = len(Xinit)
    Utot = DisToPotential(Xinit, Yinit, a, b)

    for n in range(Niter):
        i = np.random.randint(N)
        dx = delta * np.random.choice([-1, 1])
        dy = delta * np.random.choice([-1, 1])

        Xtrial = np.copy(Xinit)
        Ytrial = np.copy(Yinit)

        Xtrial[i] += dx
        Ytrial[i] += dy

        U_old = Utot[i]
        U_new_i = DisToPotential(Xtrial, Ytrial, a, b)[i]
        dU = U_new_i - U_old

        rho = np.random.rand()
        if dU < 0 or rho < np.exp(-dU / T):
            Xinit[i] = Xtrial[i]
            Yinit[i] = Ytrial[i]
            Utot[i] = U_new_i

        # Plot every 100 steps
        if not n % 100:
            plt.scatter(Xinit, Yinit, c='gray', alpha=0.5)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'after {n} iterations\nT = {T}, a = {a}, b = {b}')
            plt.grid(True)
            plt.show()

# 5. Simulation wrapper class
class SimulateParticles:
    def __init__(self, N=100, R=50):
        self.N = N
        self.R = R
        self.Xinit, self.Yinit = PlotLocations(R, N)

    def run(self, Niter=1000, a=1, b=1, T=1, delta=1):
        MoveParticle(self.Xinit, self.Yinit, self.N, Niter, a, b, T, delta)

# 6. Run the simulation
sim = SimulateParticles(N=200)
sim.run(Niter=6000, T=1)