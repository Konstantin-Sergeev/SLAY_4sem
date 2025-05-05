#ifndef QR_SOLVER_H
#define QR_SOLVER_H

#include "qr_decomposition.h"
#include <vector>
#include <stdexcept>

class QRSolver {
public:
    static std::vector<double> solve(const std::vector<std::vector<double>>& A, 
                                    const std::vector<double>& b) {
        if (A.empty() || b.empty() || A.size() != b.size()) {
            throw std::invalid_argument("Invalid input dimensions");
        }

        auto [Q, R] = QRDecomposition::decompose(A);
        
        // Вычисляем Q^T * b
        std::vector<double> Qt_b(b.size(), 0.0);
        for (size_t i = 0; i < Q.size(); ++i) {
            for (size_t j = 0; j < Q[0].size(); ++j) {
                Qt_b[j] += Q[i][j] * b[i];
            }
        }

        // Обратная подстановка для Rx = Q^Tb
        return backSubstitution(R, Qt_b);
    }

private:
    static std::vector<double> backSubstitution(const std::vector<std::vector<double>>& R,
                                               const std::vector<double>& b) {
        const size_t n = R.size();
        std::vector<double> x(n, 0.0);

        for (int i = n - 1; i >= 0; --i) {
            if (std::fabs(R[i][i]) < 1e-12) {
                throw std::runtime_error("Matrix is singular or nearly singular");
            }

            x[i] = b[i];
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= R[i][j] * x[j];
            }
            x[i] /= R[i][i];
        }

        return x;
    }
};

#endif // QR_SOLVER_H
