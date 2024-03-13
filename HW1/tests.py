import scipy.linalg as sl
import numpy as np
import criterions


def gilbert_matrix_case():
    A = np.ndarray((10, 10))
    for i in range(10):
        for j in range(10):
            A[i][j] = 1 / (i + j + 1)
    print()
    print("МАТРИЦА ГИЛЬБЕРТА:")
    print("Число обусловленности, вычисленное по спектральному критерию:", criterions.spectral_criterion(A))
    print("Число обусловленности, вычисленное по объёмному критерию:", criterions.volume_criterion(A))
    print("Число обусловленности, вычисленное по угловому критерию:", criterions.angle_criterion(A))
    b = np.array([1, 0.3, 0.45, 0.2344, 0.23, 10, 102, 10, 9, 10])
    b_var = b + (1e-2 - 1e-10)
    x = sl.solve(A, b)
    x_var = sl.solve(A, b_var)
    delta = abs(x - x_var)
    print("Погрешность при варьировании правой части на 10^-2 - 10^-10:")
    print(list(delta))


def diagonal_matrix_case():
    A = np.ndarray((10, 10))
    for i in range(10):
        A[i][i] = i
    print()
    print("ДИАГОНАЛЬНАЯ МАТРИЦА:")
    print("Число обусловленности, вычисленное по спектральному критерию:", criterions.spectral_criterion(A))
    print("Число обусловленности, вычисленное по объёмному критерию:", criterions.volume_criterion(A))
    print("Число обусловленности, вычисленное по угловому критерию:", criterions.angle_criterion(A))
    b = np.array([1 / i ** 2 for i in range(1, 11)])
    b_var = b + (1e-2 - 1e-10)
    x = sl.solve(A, b)
    x_var = sl.solve(A, b_var)
    delta = abs(x - x_var)
    print("Погрешность при варьировании правой части на 10^-2 - 10^-10:")
    print(list(delta))


def tridiagonal_matrix():
    A = np.ndarray((10, 10))
    for i in range(10):
        A[i][i] = 2
        if i - 1 >= 0:
            A[i][i - 1] = -1
        if i + 1 <= 9:
            A[i][i + 1] = -1
    print()
    print("ТРЁХДИАГОНАЛЬНАЯ МАТРИЦА:")
    print("Число обусловленности, вычисленное по спектральному критерию:", criterions.spectral_criterion(A))
    print("Число обусловленности, вычисленное по объёмному критерию:", criterions.volume_criterion(A))
    print("Число обусловленности, вычисленное по угловому критерию:", criterions.angle_criterion(A))
    b = np.array([1 / i ** 2 for i in range(1, 11)])
    b_var = b + (1e-2 - 1e-10)
    x = sl.solve(A, b)
    x_var = sl.solve(A, b_var)
    delta = abs(x - x_var)
    print("Погрешность при варьировании правой части на 10^-2 - 10^-10:")
    print(list(delta))


def regular_matrix():
    A = np.array([[1, 0, 1],
                  [2, 3, 4],
                  [1, 3, 2]])
    print()
    print("ОБЫЧНАЯ МАТРИЦА:")
    print("Число обусловленности, вычисленное по спектральному критерию:", criterions.spectral_criterion(A))
    print("Число обусловленности, вычисленное по объёмному критерию:", criterions.volume_criterion(A))
    print("Число обусловленности, вычисленное по угловому критерию:", criterions.angle_criterion(A))
    b = np.array([i for i in range(1, 4)])
    b_var = b + (1e-2 - 1e-10)
    x = sl.solve(A, b)
    x_var = sl.solve(A, b_var)
    delta = abs(x - x_var)
    print("Погрешность при варьировании правой части на 10^-2 - 10^-10:")
    print(list(delta))


gilbert_matrix_case()
diagonal_matrix_case()
tridiagonal_matrix()
regular_matrix()
