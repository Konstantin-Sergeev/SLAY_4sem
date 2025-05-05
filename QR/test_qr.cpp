#include "qr_decomposition.h"
#include "qr_solver.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

void printMatrix(const std::vector<std::vector<double>>& A) {
    for (const auto& row : A) {
        for (double val : row) {
            std::cout << std::setw(12) << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }
}

void printVector(const std::vector<double>& v) {
    for (double val : v) {
        std::cout << std::setprecision(8) << val << " ";
    }
    std::cout << std::endl;
}

bool matricesEqual(const std::vector<std::vector<double>>& A, 
                  const std::vector<std::vector<double>>& B, 
                  double epsilon = 1e-6) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) return false;
    
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            if (std::fabs(A[i][j] - B[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

void testQRDecomposition() {
    std::cout << "=== Testing QR Decomposition ===" << std::endl;
    
    const std::vector<std::vector<double>> A = {
        {12.0, -51.0, 4.0},
        {6.0, 167.0, -68.0},
        {-4.0, 24.0, -41.0}
    };
    
    std::cout << "Original matrix A:" << std::endl;
    printMatrix(A);
    
    auto [Q, R] = QRDecomposition::decompose(A);
    
    std::cout << "\nOrthogonal matrix Q:" << std::endl;
    printMatrix(Q);
    
    std::cout << "\nUpper triangular matrix R:" << std::endl;
    printMatrix(R);
    
    // Проверка QR = A
    auto QR = QRDecomposition::multiply(Q, R);
    std::cout << "\nQR product:" << std::endl;
    printMatrix(QR);
    
    if (matricesEqual(QR, A, 1e-8)) {
        std::cout << "\nSUCCESS: QR equals original matrix A (within tolerance)" << std::endl;
    } else {
        std::cout << "\nFAILURE: QR doesn't equal original matrix A" << std::endl;
    }
    
    // Проверка ортогональности Q (Q^TQ = I)
    auto Qt = Q;
    for (size_t i = 0; i < Qt.size(); ++i) {
        for (size_t j = 0; j < Qt[0].size(); ++j) {
            Qt[i][j] = Q[j][i];
        }
    }
    
    auto QtQ = QRDecomposition::multiply(Qt, Q);
    std::vector<std::vector<double>> I(Q.size(), std::vector<double>(Q.size(), 0.0));
    for (size_t i = 0; i < I.size(); ++i) {
        I[i][i] = 1.0;
    }
    
    if (matricesEqual(QtQ, I, 1e-8)) {
        std::cout << "\nSUCCESS: Q is orthogonal (Q^TQ = I)" << std::endl;
    } else {
        std::cout << "\nFAILURE: Q is not orthogonal" << std::endl;
    }
}

void testQRSystemSolving() {
    std::cout << "\n=== Testing QR System Solving ===" << std::endl;
    
    const std::vector<std::vector<double>> A = {
        {1.0, 1.0, 1.0},
        {4.0, 3.0, -1.0},
        {3.0, 5.0, 3.0}
    };
    
    const std::vector<double> b = {1.0, 6.0, 4.0};
    const std::vector<double> expected_x = {1.0, 0.5, -0.5};
    
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A);
    
    std::cout << "\nVector b:" << std::endl;
    printVector(b);
    
    std::vector<double> x = QRSolver::solve(A, b);
    
    std::cout << "\nComputed solution x:" << std::endl;
    printVector(x);
    
    std::cout << "\nExpected solution x:" << std::endl;
    printVector(expected_x);
    
    // Проверка точности решения
    bool solution_correct = true;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::fabs(x[i] - expected_x[i]) > 1e-8) {
            solution_correct = false;
            break;
        }
    }
    
    if (solution_correct) {
        std::cout << "\nSUCCESS: Computed solution matches expected solution" << std::endl;
    } else {
        std::cout << "\nFAILURE: Computed solution doesn't match expected solution" << std::endl;
    }
    
    // Проверка Ax = b
    std::vector<double> Ax(b.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            Ax[i] += A[i][j] * x[j];
        }
    }
    
    std::cout << "\nAx product:" << std::endl;
    printVector(Ax);
    
    bool system_solved = true;
    for (size_t i = 0; i < b.size(); ++i) {
        if (std::fabs(Ax[i] - b[i]) > 1e-8) {
            system_solved = false;
            break;
        }
    }
    
    if (system_solved) {
        std::cout << "\nSUCCESS: Ax equals b (system solved correctly)" << std::endl;
    } else {
        std::cout << "\nFAILURE: Ax doesn't equal b" << std::endl;
    }
}

int main() {
    testQRDecomposition();
    testQRSystemSolving();
    
    return 0;
}
