import numpy as np
from random import randint
from task import bring_system_to_equivalent_form, simple_iteration_method, seidel_method


def generate_sparse_symmetric_matrix_with_diagonal_dominance(n, occupancy_rate, max_common_value):
    """
    :param n: размер квадратной матрицы
    :param occupancy_rate: приблизительный процент заполненности разреженной матрицы
    :param max_common_value: максимально возможное значение недиагонального элемента
    :return: сгенерированная случайным образом разреженная симметричная матрица с диагональным преобладанием
    """
    A = np.zeros((n, n))
    values_count = int(n * n * occupancy_rate / 2)
    for _ in range(values_count):
        # Генерация позиции в левом нижнем углу матрицы
        i = randint(0, n - 1)
        j = randint(0, i)
        num = randint(1, max_common_value)
        A[i][j] = num
        A[j][i] = num
    # Заполнение диагонали таким образом, чтобы матрица была с диагональным преобладанием
    for k in range(n):
        A[k][k] = randint(max_common_value * n, max_common_value * n * 5)
    return A


# Исходные данные для тестовой СЛАУ
n = 250
A = generate_sparse_symmetric_matrix_with_diagonal_dominance(n, 0.1, 50)
b = [randint(0, 1000) for _ in range(n)]

eps = 1e-10
H, g = bring_system_to_equivalent_form(A, b)
x1, k1 = simple_iteration_method(H, g, eps)
x2, k2 = seidel_method(H, g, eps)

print(f"\nКоличество итераций, проведённых для решения СЛАУ {n} * {n}:")
print(f"Методом простой итерации: {k1}")
print(f"Методом Зейделя: {k2}")

x = np.linalg.solve(A, b)
print(f"\nПроверка решений, найденных с точностью до {eps}:")
print(f"Решение, полученное методом простой итерации: "
      f"{"Совпадает с библиотечным" if np.allclose(x1, x, atol=eps) else "Не совпадает с библиотечным"}")
print(f"Решение, полученное методом Зейделя: "
      f"{"Совпадает с библиотечным" if np.allclose(x2, x, atol=eps) else "Не совпадает с библиотечным"}")
