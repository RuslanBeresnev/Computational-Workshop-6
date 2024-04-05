import numpy as np
from task import print_results


def symmetric_matrix_test_case():
    A = np.array([[8.67313, 1.041039, -2.677712],
                  [1.041039, 6.586211, 0.623016],
                  [-2.677712, 0.623016, 5.225935]])
    x_0 = np.array([1, 1, 1])
    y_0 = np.array([0.3, -2, 1])
    print("\n\n\nСИММЕТРИЧЕСКАЯ МАТРИЦА:")
    print_results(A, x_0, y_0, epsilon=1e-6)


def diagonal_matrix_test_case():
    A = np.array([[8.67313, 0, 0],
                  [0, 6.586211, 0],
                  [0, 0, 5.225935]])
    x_0 = np.array([2, 1, 0.5])
    y_0 = np.array([0.567, 0, 1])
    print("\n\n\nДИАГОНАЛЬНАЯ МАТРИЦА:")
    print_results(A, x_0, y_0, epsilon=1e-6)


symmetric_matrix_test_case()
diagonal_matrix_test_case()
