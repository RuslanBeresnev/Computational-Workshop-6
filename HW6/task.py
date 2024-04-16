import numpy as np
import math


def max_element_method(A):
    """
    Получить позицию [i, j] максимального по модулю внедиагонального элемента матрицы
    """
    abs_matrix = np.abs(A).astype(np.float64)
    np.fill_diagonal(abs_matrix, -np.inf)
    return np.unravel_index(np.argmax(abs_matrix, axis=None), A.shape)


def optimal_element_method(A):
    """
    Получить позицию [i, j] максимального по модулю элемента в строке, определяющей круг Гершгорина
    максимального радиуса
    """
    max_row_index = np.argmax(np.sum(A ** 2, 1) - np.diag(A ** 2))
    max_row = np.abs(A[max_row_index]).astype(np.float64)
    max_row[max_row_index] = -np.inf
    max_column_index = np.argmax(max_row)
    return max_row_index, max_column_index


def get_rotation_matrix(A, i, j):
    """
    Построить ортогональную матрицу поворота T_i_j с направляющими косинусами и синусами в строках и столбцах i, j
    """
    x = -2 * A[i][j]
    y = A[i][i] - A[j][j]
    n = np.shape(A)[0]
    if y < 1e-6:
        cos_phi = 1 / (2 ** 0.5)
        sin_phi = 1 / (2 ** 0.5)
    else:
        cos_phi = (0.5 * (1 + abs(y) / ((x ** 2 + y ** 2) ** 0.5))) ** 0.5
        sin_phi = math.copysign(1, x * y) * abs(x) / (2 * cos_phi * ((x ** 2 + y ** 2) ** 0.5))

    T = np.eye(n)
    T[i][i] = cos_phi
    T[j][j] = cos_phi
    T[i][j] = -sin_phi
    T[j][i] = sin_phi
    return T


def get_off_diagonal_elements_sum(A):
    """
    Найти сумму квадратов всех внедиагональных элементов
    """
    n = len(A)
    result = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                result += A[i][j] ** 2
    return result


def get_eigenvalues(A, eps, zeroing_method):
    """
    Приближенно (с точностью до: sqrt(n - 1) * sqrt(eps)) вычислить все собственные числа матрицы A
    """
    n = np.shape(A)[0]
    opers = 0
    while get_off_diagonal_elements_sum(A) >= eps:
        i, j = zeroing_method(A)
        T = get_rotation_matrix(A, i, j)
        A = np.dot(np.dot(np.transpose(T), A), T)
        opers += 1
    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, opers


def get_gershgorin_circles(A):
    """
    Вычисление промежутков, в которых лежат все собственные числа матрицы (с помощью теоремы Гершгорина)
    """
    intervals = []
    n = np.shape(A)[0]
    for i in range(n):
        radius = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        intervals.append([A[i, i] - radius, A[i, i] + radius])

    intervals.sort(key=lambda interval: interval[0])
    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def is_all_eigenvalues_in_gershgorin_circles(eigenvalues, circles):
    """
    Проверка на факт попадания каждого собственного числа хотя бы в один круг Гершгорина
    """
    for eigenvalue in eigenvalues:
        for circle in circles:
            if eigenvalue < circle[0] or eigenvalue > circle[1]:
                return False
    return True


# Матрица Гильберта:
n = 20
H = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        H[i][j] = 1 / (i + j + 1)

# Точность, с которой будут найдены собственные числа матрицы
eps = 1e-3

print("\nПОЛНАЯ ПРОБЛЕМА СОБСТВЕННЫХ ЗНАЧЕНИЙ МАТРИЦЫ:")
print("Описание: найти все собственные числа матрицы с точностью до epsilon с помощью метода Якоби "
      "и двух стратегий выбора обнуляемых элементов\n")

print(f"Вычисления проводятся для матрицы Гильберта размерности {n} * {n}:\n")
exact_eigenvalues, _ = np.linalg.eig(H)
exact_eigenvalues.sort()
print("Собственные значения матрицы, найденные с помощью бибилиотеки:\n", exact_eigenvalues)

gershgorin_circles = get_gershgorin_circles(H)
print("\nКруги Гершгорина:\n", gershgorin_circles)

print("\nМЕТОД МАКСИМАЛЬНОГО ПО МОДУЛЮ НЕДИАГОНАЛЬНОГО ЭЛЕМЕНТА:")
max_method_eigenvalues, max_method_opers = get_eigenvalues(H, eps, max_element_method)
max_method_eigenvalues.sort()
print(f"Собственные числа (точность {eps}):", max_method_eigenvalues)
print("Количество операций:", max_method_opers)
print("Все ли собственные числа попали в область:", "ДА" if is_all_eigenvalues_in_gershgorin_circles(
    max_method_eigenvalues, gershgorin_circles) else "НЕТ")

print("\nМЕТОД ОПТИМАЛЬНОГО НЕДИАГОНАЛЬНОГО ЭЛЕМЕНТА:")
opt_method_eigenvalues, opt_method_opers = get_eigenvalues(H, eps, optimal_element_method)
opt_method_eigenvalues.sort()
print(f"Собственные числа (точность {eps}):", opt_method_eigenvalues)
print("Количество операций:", opt_method_opers)
print("Все ли собственные числа попали в область:", "ДА" if is_all_eigenvalues_in_gershgorin_circles(
    opt_method_eigenvalues, gershgorin_circles) else "НЕТ")
