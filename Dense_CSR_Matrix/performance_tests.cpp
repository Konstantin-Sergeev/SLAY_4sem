#include "matrix_classes.h"
#include <chrono>
#include <fstream>

template<typename Func>
double measure_time(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void run_performance_tests() {
    std::ofstream out_file("performance_results.csv");
    out_file << "Matrix Type,Size,Sparsity,Time\n";
    
    const std::vector<size_t> sizes = {100, 500, 1000, 2000, 5000};
    const std::vector<double> sparsities = {0.0, 0.5, 0.9, 0.95, 0.99};
    
    for (size_t size : sizes) {
        for (double sparsity : sparsities) {
            // Генерация случайного вектора
            std::vector<double> vec(size);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            for (size_t i = 0; i < size; ++i) {
                vec[i] = dis(gen);
            }
            
            // Тестирование плотной матрицы
            if (sparsity == 0.0) {
                auto dense_mat = DenseMatrix::random(size, size);
                double time = measure_time([&]() {
                    auto result = dense_mat * vec;
                });
                out_file << "Dense," << size << "," << 0 << "," << time << "\n";
                std::cout << "Dense " << size << "x" << size << ": " << time << "s\n";
            }
            
            // Тестирование CSR матрицы
            if (sparsity > 0.0) {
                auto csr_mat = CSRMatrix::random(size, size, sparsity);
                double time = measure_time([&]() {
                    auto result = csr_mat * vec;
                });
                out_file << "CSR," << size << "," << sparsity << "," << time << "\n";
                std::cout << "CSR " << size << "x" << size << " (sparsity " << sparsity 
                          << "): " << time << "s\n";
            }
        }
    }
    
    out_file.close();
    std::cout << "Performance tests completed. Results saved to performance_results.csv" << std::endl;
}

int main() {
    // Пример использования
    std::cout << "=== Dense Matrix Example ===" << std::endl;
    DenseMatrix dense(3, 3);
    dense(0, 0) = 1; dense(0, 1) = 2; dense(0, 2) = 3;
    dense(1, 0) = 4; dense(1, 1) = 5; dense(1, 2) = 6;
    dense(2, 0) = 7; dense(2, 1) = 8; dense(2, 2) = 9;
    
    std::vector<double> vec = {1, 2, 3};
    auto result = dense * vec;
    
    std::cout << "Dense matrix:" << std::endl;
    dense.print();
    std::cout << "Vector: ";
    for (auto v : vec) std::cout << v << " ";
    std::cout << "\nResult: ";
    for (auto r : result) std::cout << r << " ";
    std::cout << std::endl << std::endl;
    
    std::cout << "=== CSR Matrix Example ===" << std::endl;
    std::map<std::pair<size_t, size_t>, double> dok = {
        {{0, 0}, 1}, {{0, 2}, 3},
        {{1, 1}, 5},
        {{2, 0}, 7}, {{2, 2}, 9}
    };
    CSRMatrix csr(dok, 3, 3);
    result = csr * vec;
    
    csr.print_info();
    std::cout << "Vector: ";
    for (auto v : vec) std::cout << v << " ";
    std::cout << "\nResult: ";
    for (auto r : result) std::cout << r << " ";
    std::cout << std::endl << std::endl;
    
    // Запуск тестов производительности
    std::cout << "=== Running Performance Tests ===" << std::endl;
    run_performance_tests();
    
    return 0;
}
