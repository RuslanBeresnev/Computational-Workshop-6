import scipy.linalg as sl


def spectral_criterion(A):
    A_inv = sl.inv(A)
    cond_s = sl.norm(A) * sl.norm(A_inv)
    return cond_s


def volume_criterion(A):
    N = A.shape[0]
    v_0 = 1
    for n in range(N):
        length = 0
        for m in range(N):
            length += A[n][m] ** 2
        v_0 *= length ** 0.5
    v = abs(sl.det(A))
    cond_v = v_0 / v
    return cond_v


def angle_criterion(A):
    N = A.shape[0]
    A_inv = sl.inv(A)
    cond_a = 0
    for n in range(N):
        a_n = 0
        c_n = 0
        for m in range(N):
            a_n += A[n][m] ** 2
            c_n += A_inv[m][n] ** 2
        a_n = a_n ** 0.5
        c_n = c_n ** 0.5
        if a_n * c_n > cond_a:
            cond_a = a_n * c_n
    return cond_a


def print_conditionality_numbers(A, Q, R):
    matrices = [("A", A), ("Q", Q), ("R", R)]
    for name, matrix in matrices:
        print(f"Для матрицы {name}:")
        print("cond_s:", spectral_criterion(matrix))
        print("cond_v:", volume_criterion(matrix))
        print("cond_a:", angle_criterion(matrix))
        print()
