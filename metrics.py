import timeit


# =========================
# Метрики
# =========================

# Невязка (||Ax - b||)
def residual(A, x, b):
    return np.linalg.norm(A @ x - b)


# Относительная погрешность (||x - x_true|| / ||x_true||)
def relative_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


# Функция для измерения времени выполнения функции
def measure(func, *args):
    return timeit.timeit(lambda: func(*args), number=1)
