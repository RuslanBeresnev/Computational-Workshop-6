import numpy as np
import matplotlib.pyplot as plt

import power_method as pm
import scalar_product_method as spm

# Исходные данные
A = np.array([[8.67313, 1.041039, -2.677712],
              [1.041039, 6.586211, 0.623016],
              [-2.677712, 0.623016, 5.225935]])
x_0 = np.array([1, 1, 1])
y_0 = np.array([0.3, -2, 1])

print("\nЧАСТИЧНАЯ ПРОБЛЕМА СОБСТВЕННЫХ ЗНАЧЕНИЙ")
print("Описание: нахождение приближенного максимального по модулю собственного числа матрицы "
      "с помощью степенного метода и метода скалярных произведений\n")

# Варьирование epsilon и нахождение для каждого такого epsilon количества итераций обоих методов
set_of_epsilon = []
set_of_pm_iters = []
set_of_spm_iters = []
for i in range(4, 1, -1):
    for j in range(1, 11):
        epsilon = j * 10 ** (-i)
        _, _, pm_iters = pm.find_max_eigenvalue_and_eigenvector(A, x_0, epsilon)
        _, _, spm_iters = spm.find_max_eigenvalue_and_eigenvector(A, x_0, y_0, epsilon)
        set_of_epsilon.append(epsilon)
        set_of_pm_iters.append(pm_iters)
        set_of_spm_iters.append(spm_iters)

# Построение сравнительного графика
plt.plot(set_of_epsilon, set_of_pm_iters, color='r')
plt.plot(set_of_epsilon, set_of_spm_iters, color='b')
plt.xlabel("Величина epsilon")
plt.ylabel("Количество итераций")
plt.title('СРАВНЕНИЕ ДВУХ МЕТОДОВ')
plt.show()


def print_results(matrix, x_0, y_0, epsilon=1e-3):
    # Библиотечное решение для максимального собственного числа и собственных векторов (NumPy)
    print("\nРЕШЕНИЯ, НАЙДЕННЫЕ С ПОМОЩЬЮ БИБЛИОТЕКИ:")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("Точное максимальное по модулю собственное число:", max(map(abs, eigenvalues)))
    print("Столбцы собственных векторов:\n", eigenvectors)

    # Решения, полученные обоими методами на переданной матрице
    pm_eigenvalue, pm_eigenvector, _ = pm.find_max_eigenvalue_and_eigenvector(matrix, x_0, epsilon)
    spm_eigenvalue, spm_eigenvector, _ = spm.find_max_eigenvalue_and_eigenvector(matrix, x_0, y_0, epsilon)
    pm_eigenvector /= np.linalg.norm(pm_eigenvector)
    spm_eigenvector /= np.linalg.norm(spm_eigenvector)
    print("\nСТЕПЕННОЙ МЕТОД:")
    print(f"Приближенное (с точностью до {epsilon}) максимальное по модулю собственное число:", pm_eigenvalue)
    print(f"Приближенный (с точностью до {epsilon}) собственный вектор\n", pm_eigenvector)
    print("\nМЕТОД СКАЛЯРНЫХ ПРОИЗВЕДЕНИЙ:")
    print(f"Приближенное (с точностью до {epsilon}) максимальное по модулю собственное число:", spm_eigenvalue)
    print(f"Приближенный (с точностью до {epsilon}) собственный вектор\n", spm_eigenvector)


print_results(A, x_0, y_0, epsilon=1e-6)
