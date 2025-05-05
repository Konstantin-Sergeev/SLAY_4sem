#ifndef QR_DECOMPOSITION_H
#define QR_DECOMPOSITION_H

#include <vector>
#include <cmath>
#include <stdexcept>

class QRDecomposition {
public:
    static std::pair<std::vector<std::vector<double>>, 
                    std::vector<std::vector<double>>> decompose(const std::vector<std::vector<double>>& A) {
        if (A.empty() || A[0].empty()) {
            throw std::invalid_argument("Matrix is empty");
        }

        const size_t m = A.size();
        const size_t n = A[0].size();
        
        std::vector<std::vector<double>> Q(m, std::vector<double>(m, 0.0));
        std::vector<std::vector<double>> R = A;

        // Инициализация Q как единичной матрицы
        for (size_t i = 0; i < m; ++i) {
            Q[i][i] = 1.0;
        }

        for (size_t k = 0; k < std::min(m, n); ++k) {
            // Вычисление вектора Хаусхолдера
            double norm = 0.0;
            for (size_t i = k; i < m; ++i) {
                norm += R[i][k] * R[i][k];
            }
            norm = std::sqrt(norm);

            const double sign = R[k][k] < 0 ? -1.0 : 1.0;
            const double u1 = R[k][k] + sign * norm;
            
            std::vector<double> w(m, 0.0);
            for (size_t i = k; i < m; ++i) {
                w[i] = R[i][k] / u1;
            }
            w[k] = 1.0;

            const double tau = sign * u1 / norm;

            // Применение преобразования Хаусхолдера к R
            for (size_t j = k; j < n; ++j) {
                double dot = 0.0;
                for (size_t i = k; i < m; ++i) {
                    dot += w[i] * R[i][j];
                }
                for (size_t i = k; i < m; ++i) {
                    R[i][j] -= tau * w[i] * dot;
                }
            }

            // Применение преобразования Хаусхолдера к Q
            for (size_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (size_t i = k; i < m; ++i) {
                    dot += Q[j][i] * w[i];
                }
                for (size_t i = k; i < m; ++i) {
                    Q[j][i] -= tau * dot * w[i];
                }
            }
        }

        return {Q, R};
    }

    static std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& A, 
                                                   const std::vector<std::vector<double>>& B) {
        if (A.empty() || B.empty() || A[0].size() != B.size()) {
            throw std::invalid_argument("Invalid matrix dimensions for multiplication");
        }

        const size_t m = A.size();
        const size_t n = B[0].size();
        const size_t p = B.size();

        std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < p; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }
};

#endif // QR_DECOMPOSITION_H
