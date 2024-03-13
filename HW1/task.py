import scipy.linalg as sl
import numpy as np
import criterions


# Информация о задании
print("НАХОЖДЕНИЕ ЧИСЕЛ ОБУСЛОВЛЕННОСТИ СЛАУ")
print("Описание: найти числа обусловленности для СЛАУ с квадратной матрицей с помощью различных критериев, "
      "сравнить их поведение на разных данных, а также сравнить числа обусловленности с величиной погрешности решения "
      "при небольшом изменении начальных данных")
print()

# Исходные данные
A = np.array([[1, 0.99],
              [0.99, 0.98]])
b = np.array([2, 2])

# Вывод исходных данных СЛАУ
print("Левая часть исходной СЛАУ:")
print(A)
print()
print("Правая часть исходной СЛАУ:")
print(b)
print()

# Решение исходной системы
x = sl.solve(A, b)
print("Решение исходной системы:", x)

# Проварьированные данные
A_var = A + (1e-2 - 1e-10)
b_var = b + (1e-2 - 1e-10)

# Решение проварьированной системы
x_var = sl.solve(A_var, b_var)
print("Решение проварьированной системы:", x_var)

# Вычисление погрешности решения
delta_x = abs(x - x_var)
print("Погрешность решения:", delta_x)
print()

# Нахождение чисел обусловленности
print("Числа обусловленности, найденные с помощью спектрального, объёмного и углового критериев:")
cond_s = criterions.spectral_criterion(A)
print("cond_s =", cond_s, "- плохая обусловленность") if cond_s > 1e4 else print("cond_s =", cond_s,
                                                                                 "- хорошая обусловленность")
cond_v = criterions.volume_criterion(A)
print("cond_v =", cond_v, "- плохая обусловленность") if cond_v > 1e4 else print("cond_v =", cond_v,
                                                                                 "- хорошая обусловленность")
cond_a = criterions.angle_criterion(A)
print("cond_a =", cond_a, "- плохая обусловленность") if cond_a > 1e4 else print("cond_a =", cond_a,
                                                                                 "- хорошая обусловленность")
