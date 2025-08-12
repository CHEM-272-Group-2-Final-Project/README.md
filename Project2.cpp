#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
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
    double choice_pm() { return uni01(gen) < 0.5 ? -1.0 : 1.0; }
    int randint(int low, int high) {
        std::uniform_int_distribution<int> d(low, high - 1);
        return d(gen);
    }
};

std::pair<std::vector<double>, std::vector<double>>
initial_positions(double radius, int n, RNG& rng) {
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = rng.uniform(-radius, radius);
        y[i] = rng.uniform(-radius, radius);
    }
    return {x, y};
}

std::vector<double> initial_masses(int n, double low, double high, RNG& rng) {
    std::vector<double> m(n);
    for (int i = 0; i < n; i++) m[i] = rng.uniform(low, high);
    return m;
}

std::vector<std::vector<double>> lennard_jones_matrix(
    const std::vector<std::vector<double>>& dx,
    const std::vector<std::vector<double>>& dy,
    double a, double b) {
    int n = dx.size();
    std::vector<std::vector<double>> phi(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double r2 = dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j];
            phi[i][j] = (a / std::pow(r2, 6)) - (b / std::pow(r2, 3));
        }
    }
    return phi;
}

std::vector<double> total_potential_per_particle(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double a, double b) {
    int n = x.size();
    std::vector<std::vector<double>> dx(n, std::vector<double>(n));
    std::vector<std::vector<double>> dy(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dx[i][j] = x[i] - x[j];
            dy[i][j] = y[i] - y[j];
        }
    }
    auto phi = lennard_jones_matrix(dx, dy, a, b);
    std::vector<double> U(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += phi[i][j];
        U[i] = sum;
    }
    return U;
}

void metropolis_random_scan(
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& U,
    double a, double b, double T, double delta, double box_half, RNG& rng) {
    int n = x.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng.gen);
    for (int i : idx) {
        double dx = delta * rng.choice_pm();
        double dy = delta * rng.choice_pm();
        std::vector<double> xt = x;
        std::vector<double> yt = y;
        xt[i] = std::max(-box_half, std::min(box_half, xt[i] + dx));
        yt[i] = std::max(-box_half, std::min(box_half, yt[i] + dy));
        double U_new_i = total_potential_per_particle(xt, yt, a, b)[i];
        double dU = U_new_i - U[i];
        if (dU < 0.0 || rng.uni01(rng.gen) < std::exp(-dU / T)) {
            x[i] = xt[i];
            y[i] = yt[i];
            U[i] = U_new_i;
        }
    }
}

void metropolis_all_at_once(
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& U,
    double a, double b, double T, double delta, double box_half, RNG& rng) {
    int n = x.size();
    std::vector<double> xt(n), yt(n);
    for (int i = 0; i < n; i++) {
        xt[i] = std::max(-box_half, std::min(box_half, x[i] + delta * rng.choice_pm()));
        yt[i] = std::max(-box_half, std::min(box_half, y[i] + delta * rng.choice_pm()));
    }
    auto U_new = total_potential_per_particle(xt, yt, a, b);
    for (int i = 0; i < n; i++) {
        double dU = U_new[i] - U[i];
        if (dU < 0.0 || rng.uni01(rng.gen) < std::exp(-dU / T)) {
            x[i] = xt[i];
            y[i] = yt[i];
            U[i] = U_new[i];
        }
    }
}

void plot_particles(const std::vector<double>& x,
                    const std::vector<double>& y,
                    const std::vector<double>& mass,
                    double box_half,
                    int step, double T, double a, double b) {
    FILE* gp = POPEN("gnuplot -persist", "w");
    if (!gp) return;
    fprintf(gp, "set title 'Iteration %d  T=%.3f  a=%.3f  b=%.3f'\n", step, T, a, b);
    fprintf(gp, "set xrange [%f:%f]\n", -box_half, box_half);
    fprintf(gp, "set yrange [%f:%f]\n", -box_half, box_half);
    fprintf(gp, "set size square\n");
    fprintf(gp, "unset key\n");
    fprintf(gp, "plot '-' with points pt 7 ps variable\n");
    for (size_t i = 0; i < x.size(); i++) {
        double size = 10.0 * mass[i] * 0.1;
        fprintf(gp, "%f %f %f\n", x[i], y[i], size);
    }
    fprintf(gp, "e\n");
    PCLOSE(gp);
}

struct Simulator {
    int n;
    double radius;
    RNG rng;
    std::vector<double> x, y, mass;
    Simulator(int n_, double radius_, unsigned seed)
        : n(n_), radius(radius_), rng(seed) {
        auto pos = initial_positions(radius, n, rng);
        x = pos.first;
        y = pos.second;
        mass = initial_masses(n, 0.5, 2.0, rng);
    }
    void run(int n_iter, double a, double b, double T, double delta, double box_half,
             int plot_every, const std::string& mode) {
        auto U = total_potential_per_particle(x, y, a, b);
        for (int step = 0; step <= n_iter; step++) {
            if (mode == "all_at_once")
                metropolis_all_at_once(x, y, U, a, b, T, delta, box_half, rng);
            else
                metropolis_random_scan(x, y, U, a, b, T, delta, box_half, rng);
            if (step % plot_every == 0)
                plot_particles(x, y, mass, box_half, step, T, a, b);
        }
    }
};

int main() {
    Simulator sim(200, 100.0, 42);
    sim.run(2000, 1.0, 1.0, 25.0, 1.0, 100.0, 100, "random_scan");
    return 0;
}