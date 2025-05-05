#ifndef MATRIX_CLASSES_H
#define MATRIX_CLASSES_H

#include <vector>
#include <map>
#include <random>
#include <stdexcept>
#include <iostream>

// =============================================
// 1. Класс плотной матрицы (DenseMatrix)
// =============================================

class DenseMatrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    DenseMatrix(size_t rows, size_t cols, double initial_value = 0.0);
    DenseMatrix(const std::vector<std::vector<double>>& values);
    
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    
    std::vector<double> operator*(const std::vector<double>& vec) const;
    
    static DenseMatrix random(size_t rows, size_t cols, double min = 0.0, double max = 1.0);
    void print() const;
};

// Реализация методов DenseMatrix
DenseMatrix::DenseMatrix(size_t rows, size_t cols, double initial_value) 
    : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, initial_value));
}

DenseMatrix::DenseMatrix(const std::vector<std::vector<double>>& values) {
    if (values.empty() || values[0].empty()) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    rows = values.size();
    cols = values[0].size();
    data = values;
    
    for (const auto& row : values) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same length");
        }
    }
}

double& DenseMatrix::operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[i][j];
}

const double& DenseMatrix::operator()(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data[i][j];
}

std::vector<double> DenseMatrix::operator*(const std::vector<double>& vec) const {
    if (vec.size() != cols) {
        throw std::invalid_argument("Vector size must match matrix columns");
    }
    
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += data[i][j] * vec[j];
        }
    }
    return result;
}

DenseMatrix DenseMatrix::random(size_t rows, size_t cols, double min, double max) {
    DenseMatrix mat(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = dis(gen);
        }
    }
    return mat;
}

void DenseMatrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// =============================================
// 2. Класс CSR матрицы (CSRMatrix)
// =============================================

class CSRMatrix {
private:
    std::vector<double> values;
    std::vector<int> columns;
    std::vector<int> row_ptr;
    size_t rows;
    size_t cols;
    double sparsity;

public:
    CSRMatrix(const std::map<std::pair<size_t, size_t>, double>& dok, 
              size_t rows, size_t cols);
    
    std::vector<double> operator*(const std::vector<double>& vec) const;
    
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    double get_sparsity() const { return sparsity; }
    
    static CSRMatrix random(size_t rows, size_t cols, double sparsity = 0.9);
    void print_info() const;
};

// Реализация методов CSRMatrix
CSRMatrix::CSRMatrix(const std::map<std::pair<size_t, size_t>, double>& dok, 
                     size_t rows, size_t cols) : rows(rows), cols(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    
    row_ptr.resize(rows + 1, 0);
    
    for (const auto& entry : dok) {
        size_t i = entry.first.first;
        if (i >= rows) {
            throw std::out_of_range("Row index out of range");
        }
        row_ptr[i + 1]++;
    }
    
    for (size_t i = 1; i <= rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }
    
    values.resize(row_ptr.back());
    columns.resize(row_ptr.back());
    
    std::vector<int> current_pos(rows, 0);
    
    for (const auto& entry : dok) {
        size_t i = entry.first.first;
        size_t j = entry.first.second;
        if (j >= cols) {
            throw std::out_of_range("Column index out of range");
        }
        
        int pos = row_ptr[i] + current_pos[i];
        values[pos] = entry.second;
        columns[pos] = j;
        current_pos[i]++;
    }
    
    sparsity = 1.0 - static_cast<double>(values.size()) / (rows * cols);
}

std::vector<double> CSRMatrix::operator*(const std::vector<double>& vec) const {
    if (vec.size() != cols) {
        throw std::invalid_argument("Vector size must match matrix columns");
    }
    
    std::vector<double> result(rows, 0.0);
    
    for (size_t i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            result[i] += values[j] * vec[columns[j]];
        }
    }
    
    return result;
}

CSRMatrix CSRMatrix::random(size_t rows, size_t cols, double sparsity) {
    std::map<std::pair<size_t, size_t>, double> dok;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::bernoulli_distribution bern(1.0 - sparsity);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (bern(gen)) {
                dok[{i, j}] = dis(gen);
            }
        }
    }
    
    CSRMatrix mat(dok, rows, cols);
    return mat;
}

void CSRMatrix::print_info() const {
    std::cout << "CSR Matrix " << rows << "x" << cols 
              << " with " << values.size() << " non-zero elements ("
              << (100.0 * sparsity) << "% sparsity)" << std::endl;
}

// =============================================
// 3. Векторные операции
// =============================================

std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition");
    }
    
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction");
    }
    
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for dot product");
    }
    
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<double> operator*(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = scalar * vec[i];
    }
    return result;
}

std::vector<double> operator*(const std::vector<double>& vec, double scalar) {
    return scalar * vec;
}

#endif // MATRIX_CLASSES_H
