#include "tridiagonal_solver.h"
#include <iostream>
#include <vector>
#include <cmath>

bool testTridiagonalSolver() {
    // Тестовая система:
    // [ 2  1  0 ] [x0]   [ 4 ]
    // [ 1  2  1 ] [x1] = [ 8 ]
    // [ 0  1  2 ] [x2]   [ 8 ]
    // Точное решение: [1, 2, 3]

    std::vector<double> a = { 0.0, 1.0, 1.0 }; // нижняя диагональ (a[0] не используется)
    std::vector<double> b = { 2.0, 2.0, 2.0 }; // главная диагональ
    std::vector<double> c = { 1.0, 1.0, 0.0 }; // верхняя диагональ (c[n-1] не используется)
    std::vector<double> d = { 4.0, 8.0, 8.0 }; // правая часть

    std::vector<double> x = TridiagonalSolver::solve(a, b, c, d);

    // Проверка решения с допустимой погрешностью
    const double epsilon = 1e-10;
    if (std::abs(x[0] - 1.0) > epsilon ||
        std::abs(x[1] - 2.0) > epsilon ||
        std::abs(x[2] - 3.0) > epsilon) {
        std::cerr << "Test failed. Solution: ["
            << x[0] << ", " << x[1] << ", " << x[2]
            << "], expected [1, 2, 3]" << std::endl;
        return false;
    }

    std::cout << "Test passed. Solution: ["
        << x[0] << ", " << x[1] << ", " << x[2]
        << "]" << std::endl;
    return true;
}

int main() {
    return testTridiagonalSolver() ? 0 : 1;
}
