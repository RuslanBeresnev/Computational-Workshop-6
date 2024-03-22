import numpy as np
import criterions as cr
from task import rotations_method


def poorly_conditioned_matrix_case():
    A = np.array([[1, 0.99],
                  [0.99, 0.98]])
    b = np.array([2, 2])
    x, Q, R = rotations_method(A, b)
    print("\nЧИСЛА ОБУСЛОВЛЕННОСТИ ДЛЯ ПЛОХО ОБУСЛОВЛЕННОЙ МАТРИЦЫ:")
    cr.print_conditionality_numbers(A, Q, R)


def well_conditioned_matrix_case():
    A = np.array([[1, 3],
                  [5, 6]])
    b = np.array([10, 20])
    x, Q, R = rotations_method(A, b)
    print("\nЧИСЛА ОБУСЛОВЛЕННОСТИ ДЛЯ ХОРОШО ОБУСЛОВЛЕННОЙ МАТРИЦЫ:")
    cr.print_conditionality_numbers(A, Q, R)


poorly_conditioned_matrix_case()
well_conditioned_matrix_case()
