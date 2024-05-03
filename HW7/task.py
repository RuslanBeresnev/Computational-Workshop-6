import math
import decimal
import matplotlib.pyplot as plt

# Нужно решить следующее дифференциальное уравнение
# 1 * y" + (2 / x) * y' + 1 * y = 1 / x

# Граничные условия:
# y(pi / 2) = y(a) = 0
# y(pi) = y(b) = 0

# Должно получиться следующее решение: -sin(x) / x + cos(x) / x + 1 / x

# Нужно поменять коэффициенты ниже, если меняются граничные условия:
# alpha_1 * y(a) - alpha_2 * y'(a) = alpha
# beta_1 * y(b) + beta_2 * y'(b) = beta

a = math.pi / 2
b = math.pi
alpha = 0
beta = 0
alpha_1 = 1
beta_1 = 1
alpha_2 = 0
beta_2 = 0


def p(x):
    """
    Отрицательная функция при y" в исходном уравнении
    """
    return -1


def q(x):
    """
    Функция при y' в исходном уравнении
    """
    return 2 / x


def r(x):
    """
    Функция при y в исходном уравнении
    """
    return 1


def f(x):
    """
    Правая часть уравнения
    """
    return 1 / x


def y(x):
    """
    Аналитически найденная функция y
    """
    return -math.sin(x) / x + math.cos(x) / x + 1 / x


# Количество делений отрезка
n = 10
# Словарь, где ключ - это количество делений отрезка, а значение - максимальная погрешность в узле для данного разбиения
errors = {}
# Массив точек x_i, где i - индекс (от 0 до n)
x_vals = []
# Массив значений функции в точках x_i, где i - индекс (от 0 до n)
y_vals = []
y_vals_prev = []
for k in range(15):
    h = (b - a) / n
    x_vals = []
    # Словарь, где ключ - это индекс i (от 0 до n); значение - массив из двух значений s_i и t_i
    s_t_vals = {}

    s_i = 0
    t_i = 0

    # Прямая прогонка
    for i in range(n + 1):
        x_i = a + i * h
        x_vals.append(x_i)
        if i == 0:
            A_i = 0
            B_i = h * alpha_1 + alpha_2
            C_i = alpha_2
            G_i = -h * alpha
        elif i == n:
            A_i = beta_2
            B_i = h * beta_1 + beta_2
            C_i = 0
            G_i = -h * beta
        else:
            A_i = -p(x_i) - q(x_i) * h / 2
            C_i = -p(x_i) + q(x_i) * h / 2
            B_i = A_i + C_i - h ** 2 * r(x_i)
            G_i = h ** 2 * f(x_i)

        if i == 0:
            s_i = C_i / B_i
            t_i = -G_i / B_i
        else:
            s_i_prev = s_i
            t_i_prev = t_i
            s_i = C_i / (B_i - A_i * s_i_prev)
            t_i = (A_i * t_i_prev - G_i) / (B_i - A_i * s_i_prev)
        s_t_vals[i] = [s_i, t_i]

    y_vals = []
    y_i = 0

    # Обратная прогонка
    for i in range(n, -1, -1):
        s_i = s_t_vals[i][0]
        t_i = s_t_vals[i][1]
        if i == n:
            y_i = t_i
        else:
            y_i = s_i * y_i + t_i
        y_vals.insert(0, y_i)

    # Начинаем вычисления погрешностей только на второй итерации
    if len(y_vals_prev) == 0:
        n *= 2
        y_vals_prev = y_vals.copy()
        continue

    # Главные члены погрешности для каждого узла сетки
    R_vals = []
    for i in range(0, n + 1, 2):
        R_i = abs(y_vals[i] - y_vals_prev[i // 2])
        R_vals.append(R_i)
    errors[n] = max(R_vals)

    n *= 2
    y_vals_prev = y_vals.copy()

print("\nРЕШЕНИЕ КРАЕВОЙ ЗАДАЧИ СЕТОЧНЫМ МЕТОДОМ")
print("\nНужно решить дифференциальное уравнение:")
print("1 * y\" + (2 / x) * y' + 1 * y = 1 / x")
print("\nГраничные условия:")
print("y(pi / 2) = y(a) = 0")
print("y(pi) = y(b) = 0")

# Печать количества делений отрезка и соответствующей ему погрешности
print("\nN\t|\tError")
for n, err in errors.items():
    print(f"{n}\t\t{format(decimal.Decimal(err), '.2e')}")

# Печать значений найденного приближенного значения функции y
print("\nX\t\t|\t\tY")
for i in range(0, n + 1, 10000):
    print(f"{format(decimal.Decimal(x_vals[i]), '.2e')}\t\t{format(decimal.Decimal(y_vals[i]), '.2e')}")
print(f"{format(decimal.Decimal(x_vals[n]), '.2e')}\t\t{format(decimal.Decimal(y_vals[n]), '.2e')}")

# Построение графика зависимости погрешности нахождения функции от количества делений отрезка
plt.plot(list(errors.keys()), list(errors.values()), color='r')
plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlabel("Количество делений отрезка")
plt.ylabel("Погрешность нахождения функции")
plt.title("ЗАВИСИМОСТЬ ПОГРЕШНОСТИ ОТ КОЛИЧЕСТВА ДЕЛЕНИЙ")
plt.show()

# Построение полученного приближения функции y
plt.plot(x_vals, y_vals, color='r')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("ПОЛУЧЕННОЕ ПРИБЛИЖЕНИЕ ФУНКЦИИ Y")
plt.show()

# Построение точного графика функции y
y_ex_vals = [y(x_vals[i]) for i in range(n + 1)]
plt.plot(x_vals, y_ex_vals, color='g')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("ТОЧНЫЕ ЗНАЧЕНИЯ ФУНКЦИИ Y")
plt.show()
