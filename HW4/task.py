import numpy as np


def bring_system_to_equivalent_form(A, b):
    """
    Приведение системы Ax = b к виду x = Hx + g
    """
    n = A.shape[0]
    H = np.zeros((n, n))
    g = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i][j] = 0
            else:
                H[i][j] = -(A[i][j] / A[i][i])
        g[i] = b[i] / A[i][i]
    return H, g


def simple_iteration_method(H, g, eps):
    """
    МЕТОД ПРОСТОЙ ИТЕРАЦИИ (нахождение решения системы с точностью до epsilon)
    """
    n = H.shape[0]
    evaluation_const = np.linalg.norm(H) / (1 - np.linalg.norm(H))
    x_k = np.zeros(n)
    k = 1
    while True:
        x_k_next = np.dot(H, x_k) + g
        # Апостериорная оценка:
        if evaluation_const * np.linalg.norm(x_k_next - x_k) < eps:
            break
        x_k = x_k_next.copy()
        k += 1
    return x_k_next, k


def seidel_method(H, g, eps):
    """
    МЕТОД ЗЕЙДЕЛЯ (нахождение решения системы с точностью до epsilon)
    """
    # Разделение матрицы "H" на левую (нижнетреугольную) и правую (верхнетреугольную):
    n = H.shape[0]
    H_left = np.zeros((n, n))
    H_right = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                H_left[i][j] = H[i][j]
            else:
                H_right[i][j] = H[i][j]

    # Нахождение решения
    evaluation_const = np.linalg.norm(H) / (1 - np.linalg.norm(H))
    x_k = np.zeros(n)
    k = 1
    E_minus_H_left_inv = np.linalg.inv(np.identity(n) - H_left)
    while True:
        x_k_next = np.dot(np.dot(E_minus_H_left_inv, H_right), x_k) + np.dot(E_minus_H_left_inv, g)
        # Апостериорная оценка:
        if evaluation_const * np.linalg.norm(x_k_next - x_k) < eps:
            break
        x_k = x_k_next.copy()
        k += 1
    return x_k_next, k


print("\nИТЕРАЦИОННЫЕ МЕТОДЫ ДЛЯ РЕШЕНИЯ СЛАУ:")
print("Описание: реализация метода простой итерации и метода Зейделя для поиска решения системы Ax = b "
      "с точностью до epsilon, а затем сравнение количества итераций обоих методов\n")

A = np.array([[30, 0, 0, 0, 0],
              [0, 21, 0, 3, 4],
              [0, 0, 13, 0, 0],
              [0, 3, 0, 4, 0],
              [0, 4, 0, 0, 67]])
b = np.array([5, 10, 15, 20, 25])

print("Исходная матрица A:\n", A)
print("Правая часть b:", b)

print("\nТочное решение:", np.linalg.solve(A, b))

H, g = bring_system_to_equivalent_form(A, b)
for n in range(15):
    eps = 10 ** (-n)
    iterated_solution, k1 = simple_iteration_method(H, g, eps)
    seidel_solution, k2 = seidel_method(H, g, eps)
    print(f"\n\n\nРешение СЛАУ, найденное с точностью до epsilon = {eps}:")
    print("\nМетод простой итерации:", iterated_solution)
    print("Количество итераций:", k1)
    print("\nМетод Зейделя:", seidel_solution)
    print("Количество итераций:", k2)
