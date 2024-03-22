import numpy as np
import copy
import criterions as cr

# Начальные данные
A = np.array([[1, 0.99],
              [0.99, 0.98]])
b = np.array([2, 2])
n = A.shape[0]

print("\nМЕТОД ВРАЩЕНИЙ ДЛЯ НАХОЖДЕНИЯ РЕШЕНИЯ СЛАУ:")
print("Задача: решить систему уравнений Ax = b\n")
print("A:\n", A)
print("b:", b)


def get_rotation_matrix(source, i, j):
    """
    Вернуть матрицу поворота в плоскости ij для вектор-столбца i матрицы source
    """
    T_i_j = np.eye(n)
    vector_len = (source[i][i] ** 2 + source[i][j] ** 2) ** 0.5
    cos_phi = source[i][i] / vector_len
    sin_phi = -source[i][j] / vector_len
    T_i_j[i][i] = cos_phi
    T_i_j[j][j] = cos_phi
    T_i_j[i][j] = -sin_phi
    T_i_j[j][i] = sin_phi
    return T_i_j


def reverse_gaussian_method(R, b_new):
    """
    Найти решение для верхнетреугольной матрицы с помощью обратного прохода методом Гаусса
    """
    for j in range(n - 1, -1, -1):
        b_new[j] /= R[j][j]
        for i in range(j - 1, -1, -1):
            b_new[i] -= b_new[j] * R[i][j]


def rotations_method(A, b):
    """
    Решение СЛАУ методом вращений: Ax = b ==> QRx = b ==> Rx = Q^(-1)b ==> Rx = b_new
    Возвращает найденное решение X, а также матрицы разложения Q, R
    """
    # Преобразование исходной матрицы A к верхнетреугольной матрице R домножением слева на T_i_j
    R = copy.deepcopy(A)
    Q = np.eye(n)
    b_new = b.copy()
    for i in range(n - 1):
        for j in range(i + 1, n):
            T_i_j = get_rotation_matrix(R, i, j)
            R = np.dot(T_i_j, R)
            b_new = np.dot(T_i_j, b_new)
            Q = np.dot(Q, np.transpose(T_i_j))
    # Нахождение решения для упрощённой системы уравнений
    reverse_gaussian_method(R, b_new)
    x = b_new
    return x, Q, R


x, Q, R = rotations_method(A, b)

print("x:", x)
print("\nОртогональная матрица Q:\n", Q)
print("Верхнетреугольная матрица R:\n", R)

print("\n\nПРОВЕРКА:")
print("Исходное уравнение Ax ?= b:\n", np.dot(A, x), "?=", b)
print("Произведение QR:\n", np.dot(Q, R))

print("\n\nЧИСЛА ОБУСЛОВЛЕННОСТИ:")
cr.print_conditionality_numbers(A, Q, R)
