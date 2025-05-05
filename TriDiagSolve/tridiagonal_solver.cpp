#include "tridiagonal_solver.h"
#include <vector>
#include <stdexcept>

std::vector<double> TridiagonalSolver::solve(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {

    size_t n = b.size();

    // Проверка размеров входных данных
    if (a.size() != n || c.size() != n || d.size() != n) {
        throw std::invalid_argument("Invalid input vectors sizes");
    }
    if (n == 0) {
        return {};
    }

    std::vector<double> c_prime(n, 0.0);
    std::vector<double> d_prime(n, 0.0);
    std::vector<double> x(n, 0.0);

    // Прямой ход
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (size_t i = 1; i < n; ++i) {
        double denominator = b[i] - a[i] * c_prime[i - 1];
        if (denominator == 0.0) {
            throw std::runtime_error("Zero denominator encountered during forward sweep");
        }

        if (i < n - 1) {
            c_prime[i] = c[i] / denominator;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denominator;
    }

    // Обратный ход
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    return x;
}

