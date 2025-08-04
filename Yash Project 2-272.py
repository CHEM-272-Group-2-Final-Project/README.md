import numpy as np
import matplotlib.pyplot as plt
#different particle size different #R not r2

def PlotLocation(L=10, N=100):
    Xinit = np.random.uniform(0, L, (N,))
    Yinit = np.random.uniform(0, L, (N,))
    return Xinit, Yinit

def Potential(Dx, Dy, a=1, b=1, N=100):
    rsquared = Dx**2 + Dy**2
    Eye = np.eye(N)
    rsquared = rsquared + Eye  # avoiding division by zero
    r = np.sqrt(rsquared)
    phi = a / r**12 - b / r**6
    phi = phi * (1 - Eye)  # zeroing out diagonal
    return phi


def DisToPotential(X, Y, a, b, N, Eye, Ones):
    Xmatrix = np.tile(X, (N, 1))
    Ymatrix = np.tile(Y, (N, 1))
    Dx = Xmatrix - Xmatrix.T
    Dy = Ymatrix - Ymatrix.T
    Phi = Potential(Dx, Dy, a, b, N)
    Utot = np.dot(Phi, Ones)
    return Utot

def MoveParticle(X, Y, Utot, a=1, b=1, T=1, delta=0.01):
    N = len(X)
    Eye = np.eye(N)
    Ones = np.ones(N)

    dx = delta * np.random.choice([-1, 1], N)
    dy = delta * np.random.choice([-1, 1], N)

    Xmove = X + dx
    Ymove = Y + dy

    Unew = DisToPotential(Xmove, Ymove, a, b, N, Eye, Ones)

    for i in range(N):
        change = Unew[i] - Utot[i]
        rho = np.random.rand()

        if change < 0 or rho < np.exp(-change / T):
            X[i] = Xmove[i]
            Y[i] = Ymove[i]
            Utot[i] = Unew[i]

    return X, Y, Utot


class SimulateParticles:
    def __init__(self, N=100, L=100):
        self.N = N
        self.L = L
        self.X, self.Y = PlotLocation(L, N)
        self.Eye = np.eye(N)
        self.Ones = np.ones(N)
        self.sizes = np.random.uniform(10, 50, self.N)

    def run(self, Niter=1000, a=1, b=1, T=1, delta=0.01):
        Utot = DisToPotential(self.X, self.Y, a, b, self.N, self.Eye, self.Ones)

        for step in range(Niter):
            self.X, self.Y, Utot = MoveParticle(self.X, self.Y, Utot, a, b, T, delta)

            if step % 100 == 0:
                plt.scatter(self.X, self.Y, color='black', s=self.sizes, alpha=0.3)
                plt.title(f'Iteration {step}')
                plt.xlim(0, self.L)
                plt.ylim(0, self.L)
                plt.show()

sim = SimulateParticles(N=150)
sim.run(Niter=2500, T=1)