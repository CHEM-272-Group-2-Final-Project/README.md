"""

272_Project 2
Seungho Yoo

"""


import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1) Current locations
# -------------------------------
def PlotLocations(radius=100.0, n_particles=200, rng=None):
    """
    Return initial x,y positions uniformly in [-radius, radius].
    Shapes: (N,), (N,)
    """
    rng = np.random.default_rng(rng)
    x = rng.uniform(-radius, radius, size=n_particles)
    y = rng.uniform(-radius, radius, size=n_particles)
    return x, y


# -------------------------------
# 2) Lennard-Jones potential (pairwise matrix -> not summed)
# -------------------------------
def Potential(Dx, Dy, a=1.0, b=1.0):
    """
    Given pairwise differences Dx, Dy (NxN), return Phi (NxN) with
    Phi[i,j] = a/r_ij^12 - b/r_ij^6  and Phi[i,i] = 0.
    Uses r^2 to avoid unnecessary sqrt and protects diagonal.
    """
    r2 = Dx**2 + Dy**2
    # protect diagonal from division by zero
    n = r2.shape[0]
    r2 = r2 + np.eye(n)
    Phi = (a / (r2**6)) - (b / (r2**3))
    np.fill_diagonal(Phi, 0.0)
    return Phi


# -------------------------------
# 3) Convert positions to total potential per particle
# -------------------------------
def DistToPotential(x, y, a=1.0, b=1.0):
    """
    From position vectors x,y (N,), compute per-particle total potential:
    U_i = sum_j Phi[i,j]
    Returns U (N,)
    """
    n = x.size
    X = np.tile(x, (n, 1))
    Y = np.tile(y, (n, 1))
    Dx = X - X.T
    Dy = Y - Y.T
    Phi = Potential(Dx, Dy, a=a, b=b)
    U = Phi @ np.ones(n)
    return U


# -------------------------------
# 4) Metropolis move step
# -------------------------------
def MoveParticle(x, y, n_iter=1000, a=1.0, b=1.0, T=1.0, delta=1.0,
                 box_halfwidth=100.0, plot_iters=(0, 2500, 5600), rng=None):
    """
    In-place Metropolis updates:
      - Random ±delta step per particle (random-scan sweep)
      - Hard-wall box confinement [-box_halfwidth, +box_halfwidth]
      - Accept if dU < 0 or with prob exp(-dU/T)

    Returns (x, y) after n_iter iterations.
    """
    rng = np.random.default_rng(rng)
    n = x.size

    # initial energies
    U = DistToPotential(x, y, a=a, b=b)

    for it in range(n_iter):
        # one "sweep": try to move each particle once
        for i in rng.permutation(n):
            dx = delta * rng.choice((-1.0, 1.0))
            dy = delta * rng.choice((-1.0, 1.0))

            x_trial = x.copy()
            y_trial = y.copy()
            x_trial[i] = np.clip(x_trial[i] + dx, -box_halfwidth, box_halfwidth)
            y_trial[i] = np.clip(y_trial[i] + dy, -box_halfwidth, box_halfwidth)

            # energy difference for particle i (recompute per-particle totals)
            U_new_i = DistToPotential(x_trial, y_trial, a=a, b=b)[i]
            dU = U_new_i - U[i]

            if (dU < 0.0) or (rng.random() < np.exp(-dU / T)):
                # accept: commit trial position and update that particle's energy
                x[i] = x_trial[i]
                y[i] = y_trial[i]
                U[i] = U_new_i

        # optional plotting
        if it in plot_iters:
            plt.figure(figsize=(5, 5))
            plt.scatter(x, y, s=10, alpha=0.6)
            plt.xlim(-box_halfwidth, box_halfwidth)
            plt.ylim(-box_halfwidth, box_halfwidth)
            plt.gca().set_aspect('equal', 'box')
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"After {it} iterations  (T={T}, a={a}, b={b})")
            plt.tight_layout()
            plt.show()

    return x, y


# -------------------------------
# 5) Simulation wrapper
# -------------------------------
class SimulateParticles:
    """
    Encapsulates the LJ + Metropolis simulation.
    """
    def __init__(self, n_particles=200, radius=100.0, seed=None):
        self.n_particles = n_particles
        self.radius = radius
        self.seed = seed
        self.x, self.y = PlotLocations(radius=self.radius,
                                       n_particles=self.n_particles,
                                       rng=self.seed)

    def run(self, n_iter=6000, a=1.0, b=1.0, T=1.0, delta=1.0,
            box_halfwidth=100.0, plot_iters=(0, 2500, 5600)):
        """
        Run the simulation in-place. Returns final (x,y).
        """
        self.x, self.y = MoveParticle(self.x, self.y,
                                      n_iter=n_iter,
                                      a=a, b=b, T=T, delta=delta,
                                      box_halfwidth=box_halfwidth,
                                      plot_iters=plot_iters,
                                      rng=self.seed)
        return self.x, self.y


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    sim = SimulateParticles(n_particles=200, radius=100.0, seed=42)
    sim.run(n_iter=6000, a=1.0, b=1.0, T=25.0, delta=1.0,
            box_halfwidth=100.0, plot_iters=(0, 2500, 5600))