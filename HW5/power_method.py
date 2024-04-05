import math
import numpy as np


def calculate_eigenvalue(x_k, x_k_next):
    return math.sqrt(np.dot(x_k_next, x_k_next) / np.dot(x_k, x_k))


def calculate_posterior_error(x_k, x_k_next, eigenvalue):
    return np.linalg.norm(x_k_next - eigenvalue * x_k) / np.linalg.norm(x_k)


def find_max_eigenvalue_and_eigenvector(A, x_k, epsilon):
    iters = 0
    while True:
        iters += 1
        x_k_next = np.dot(A, x_k)
        eigenvalue = calculate_eigenvalue(x_k, x_k_next)
        posterior_error = calculate_posterior_error(x_k, x_k_next, eigenvalue)
        # Нормируем вектор, если его норма стала слишком большой (чтобы не было переполнения и потерь точности)
        if np.linalg.norm(x_k_next) > 1e3:
            x_k_next /= np.linalg.norm(x_k_next)
        # Проводим вычисления до достижения точности epsilon
        if posterior_error < epsilon:
            return eigenvalue, x_k_next, iters
        x_k = x_k_next.copy()
