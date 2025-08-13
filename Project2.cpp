#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct RNG {
    std::mt19937 gen;
    std::uniform_real_distribution<double> uni01{0.0, 1.0};
    explicit RNG(unsigned seed) : gen(seed) {}
    double uniform(double a, double b) {
        std::uniform_real_distribution<double> d(a, b);
        return d(gen);
    }
    double choice_pm() { return uni01(gen) < 0.5 ? -1.0 : 1.0; }
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
    int n = static_cast<int>(dx.size());
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
    int n = static_cast<int>(x.size());
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
    int n = static_cast<int>(x.size());
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
    int n = static_cast<int>(x.size());
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

void save_csv(const std::string& filename,
              const std::vector<double>& x,
              const std::vector<double>& y,
              const std::vector<double>& mass,
              int step, double T, double a, double b) {
    std::ofstream file(filename);
    file << "step,x,y,mass,T,a,b\n";
    for (size_t i = 0; i < x.size(); i++) {
        file << step << "," << x[i] << "," << y[i] << "," << mass[i]
             << "," << T << "," << a << "," << b << "\n";
    }
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
            if (step % plot_every == 0) {
                std::string fname = "particles_step_" + std::to_string(step) + ".csv";
                save_csv(fname, x, y, mass, step, T, a, b);
            }
        }
    }
};

int main() {
    Simulator sim(200, 100.0, 42);
    sim.run(4000, 1.0, 1.0, 25.0, 1.0, 100.0, 500, "random_scan");
    return 0;
}