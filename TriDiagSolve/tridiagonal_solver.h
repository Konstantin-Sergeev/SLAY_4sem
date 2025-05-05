#ifndef TRIDIAGONAL_SOLVER_H
#define TRIDIAGONAL_SOLVER_H

#include <vector>

class TridiagonalSolver {
public:
    // Решение системы с трёхдиагональной матрицей
    // a - нижняя диагональ (размер N-1, a[0] не используется)
    // b - главная диагональ (размер N)
    // c - верхняя диагональ (размер N-1, c[N-1] не используется)
    // d - правая часть (размер N)
    // Возвращает решение x (размер N)
    static std::vector<double> solve(
        const std::vector<double>& a,
        const std::vector<double>& b,
        const std::vector<double>& c,
        const std::vector<double>& d);
};

#endif // TRIDIAGONAL_SOLVER_H
