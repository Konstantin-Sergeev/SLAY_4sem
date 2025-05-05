cmake_minimum_required(VERSION 3.10)
project(TridiagonalSolver)

# Установка стандарта C++ (используем C++11 или новее)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Включаем тестирование
enable_testing()

# Основная библиотека
add_library(tridiagonal_solver STATIC 
    tridiagonal_solver.cpp
)

# Тестовый исполняемый файл
add_executable(test_tridiagonal_solver 
    test_tridiagonal_solver.cpp
)

# Связываем тесты с библиотекой
target_link_libraries(test_tridiagonal_solver 
    tridiagonal_solver
)

# Добавляем тест в CTest
add_test(NAME test_tridiagonal_solver 
    COMMAND test_tridiagonal_solver
)
