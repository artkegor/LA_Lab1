import numpy as np

# Фиксируем seed для воспроизводимости результатов
rng = np.random.default_rng(42)


# =========================
# Генерация данных
# =========================
def generate_matrix(n):
    return rng.uniform(-1, 1, (n, n))


def generate_vector(n):
    return rng.uniform(-1, 1, n)


def hilbert_matrix(n):
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
