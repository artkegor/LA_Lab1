import numpy as np


# =========================
# Метод Гаусса (без выбора)
# =========================
def gauss(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    # Прямой ход (приведение к верхнетреугольному виду)
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]

    return x


# =========================
# Метод Гаусса (с выбором)
# =========================
def gauss_pivot(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    for i in range(n):
        # Ищем строку с максимальным элементом в текущем столбце
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]

    return x


# =========================
# LU-разложение
# =========================
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Верхняя матрица
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        # Нижняя матрица
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U


# =========================
# Решение через LU
# =========================
def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i][:i], y[:i])

    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i][i + 1:], x[i + 1:])) / U[i][i]

    return x


def solve_lu(L, U, b):
    y = forward_substitution(L, b)
    return backward_substitution(U, y)
