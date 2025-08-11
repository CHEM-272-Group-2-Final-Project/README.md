// lj_metropolis_gnuplot.cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#ifdef _WIN32
  #define POPEN  _popen
  #define PCLOSE _pclose
#else
  #define POPEN  popen
  #define PCLOSE pclose
#endif

struct RNG {
    std::mt19937 gen;
    std::uniform_real_distribution<double> uni01{0.0, 1.0};
    explicit RNG(unsigned seed) : gen(seed) {}

    double uniform(double a, double b) {
        std::uniform_real_distribution<double> d(a, b);
        return d(gen);
    }
    double choice_pm() { return (uni01(gen) < 0.5) ? -1.0 : 1.0; }
};

static void plot_scatter_gnuplot(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 double box_halfwidth,
                                 int iter, double T, double a, double b)
{
    FILE* gp = POPEN("gnuplot -persist", "w");
    if (!gp) { std::cerr << "Failed to open gnuplot\n"; return; }

    // Configure plot
    std::fprintf(gp, "set term qt noenhanced\n");
    std::fprintf(gp, "set title 'After %d iterations  (T=%.3g, a=%.3g, b=%.3g)'\n", iter, T, a, b);
    std::fprintf(gp, "set size square\n");
    std::fprintf(gp, "set xrange [%g:%g]\n", -box_halfwidth, box_halfwidth);
    std::fprintf(gp, "set yrange [%g:%g]\n", -box_halfwidth, box_halfwidth);
    std::fprintf(gp, "set grid\n");
    std::fprintf(gp, "set xlabel 'x'\n");
    std::fprintf(gp, "set ylabel 'y'\n");
    std::fprintf(gp, "plot '-' using 1:2 with points pt 7 ps 0.5 lc rgb 'blue' notitle\n");

    // Stream data
    for (size_t i = 0; i < x.size(); ++i) {
        std::fprintf(gp, "%g %g\n", x[i], y[i]);
    }
    std::fprintf(gp, "e\n");
    std::fflush(gp);
    PCLOSE(gp);
}

static std::pair<std::vector<double>, std::vector<double>>
PlotLocations(double radius, int n_particles, RNG& rng)
{
    std::vector<double> x(n_particles), y(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        x[i] = rng.uniform(-radius, radius);
        y[i] = rng.uniform(-radius, radius);
    }
    return {std::move(x), std::move(y)};
}

static std::vector<std::vector<double>>
Potential(const std::vector<std::vector<double>>& Dx,
          const std::vector<std::vector<double>>& Dy,
          double a = 1.0, double b = 1.0)
{
    int n = static_cast<int>(Dx.size());
    std::vector<std::vector<double>> Phi(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue; // diagonal 0
            double r2 = Dx[i][j]*Dx[i][j] + Dy[i][j]*Dy[i][j];
            // Protect against r2=0 (shouldn’t happen due to i==j guard)
            // Same as Python: a/r^12 - b/r^6 = a/(r2^6) - b/(r2^3)
            Phi[i][j] = (a / std::pow(r2, 6)) - (b / std::pow(r2, 3));
        }
    }
    return Phi;
}

static std::vector<double>
DistToPotential(const std::vector<double>& x,
                const std::vector<double>& y,
                double a = 1.0, double b = 1.0)
{
    int n = static_cast<int>(x.size());
    std::vector<std::vector<double>> Dx(n, std::vector<double>(n));
    std::vector<std::vector<double>> Dy(n, std::vector<double>(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Dx[i][j] = x[i] - x[j];
            Dy[i][j] = y[i] - y[j];
        }
    }

    auto Phi = Potential(Dx, Dy, a, b);
    std::vector<double> U(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += Phi[i][j];
        U[i] = s;
    }
    return U;
}

static void MoveParticle(std::vector<double>& x,
                         std::vector<double>& y,
                         int n_iter, double a, double b, double T, double delta,
                         double box_halfwidth,
                         const std::vector<int>& plot_iters,
                         RNG& rng)
{
    const int n = static_cast<int>(x.size());
    std::vector<double> U = DistToPotential(x, y, a, b);

    for (int it = 0; it < n_iter; ++it) {
        // Random permutation of particle indices
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), rng.gen);

        for (int idx : indices) {
            double dx = delta * rng.choice_pm();
            double dy = delta * rng.choice_pm();

            std::vector<double> x_trial = x;
            std::vector<double> y_trial = y;

            x_trial[idx] = std::max(-box_halfwidth, std::min(box_halfwidth, x_trial[idx] + dx));
            y_trial[idx] = std::max(-box_halfwidth, std::min(box_halfwidth, y_trial[idx] + dy));

            // Compute only U_i' like Python (recompute whole U and take [i])
            double U_new_i = DistToPotential(x_trial, y_trial, a, b)[idx];
            double dU = U_new_i - U[idx];

            if ((dU < 0.0) || (rng.uni01(rng.gen) < std::exp(-dU / T))) {
                x[idx] = x_trial[idx];
                y[idx] = y_trial[idx];
                U[idx] = U_new_i;
            }
        }

        // Optional plotting at selected iterations
        if (std::find(plot_iters.begin(), plot_iters.end(), it) != plot_iters.end()) {
            plot_scatter_gnuplot(x, y, box_halfwidth, it, T, a, b);
        }
    }
}

int main() {
    // Params (mirror Python example)
    const int    n_particles   = 200;
    const double radius        = 100.0;
    const double a             = 1.0;
    const double b             = 1.0;
    const double T             = 25.0;
    const double delta         = 1.0;
    const double box_halfwidth = 100.0;
    const int    n_iter        = 6000;
    const std::vector<int> plot_iters = {0, 2500, 5600};

    RNG rng(42);

    // Avoid structured bindings to silence your warning:
    auto xy = PlotLocations(radius, n_particles, rng);
    std::vector<double> x = std::move(xy.first);
    std::vector<double> y = std::move(xy.second);

    MoveParticle(x, y, n_iter, a, b, T, delta, box_halfwidth, plot_iters, rng);

    std::cout << "Simulation complete.\n";
    return 0;
}