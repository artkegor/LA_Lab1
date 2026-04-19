import pandas as pd
import matplotlib.pyplot as plt

from data import *
from metrics import *


# =========================
# Эксперимент 1
# =========================
def experiment_single():
    sizes = [100, 200, 500]

    results = []

    for n in sizes:
        print(f"\nРазмер n = {n}")

        A = generate_matrix(n)
        b = generate_vector(n)

        t_gauss = measure(gauss, A, b)
        t_pivot = measure(gauss_pivot, A, b)

        t_lu = measure(lu_decomposition, A)
        L, U = lu_decomposition(A)
        t_solve = measure(solve_lu, L, U, b)

        results.append([n, t_gauss, t_pivot, t_lu, t_solve])

        print(f"Гаусс: {t_gauss:.6f}")
        print(f"Гаусс (с выбором): {t_pivot:.6f}")
        print(f"LU разложение: {t_lu:.6f}")
        print(f"LU решение: {t_solve:.6f}")

    df = pd.DataFrame(results, columns=[
        "Размер n",
        "Гаусс (без выбора)",
        "Гаусс (с выбором)",
        "LU разложение",
        "LU решение"
    ])

    df.to_csv("experiment1.csv", index=False)
    print("\nТаблица сохранена в experiment1.csv")


# =========================
# Эксперимент 2
# =========================
def experiment_multiple():
    n = 500
    ks = [1, 10, 100]

    A = generate_matrix(n)

    gauss_times = []
    lu_times = []

    for k in ks:
        print(f"\nk = {k}")

        Bs = [generate_vector(n) for _ in range(k)]

        t0 = timeit.timeit(lambda: [gauss_pivot(A, b) for b in Bs], number=1)

        def lu_block():
            L, U = lu_decomposition(A)
            for b in Bs:
                solve_lu(L, U, b)

        t1 = timeit.timeit(lu_block, number=1)

        gauss_times.append(t0)
        lu_times.append(t1)

        print(f"Гаусс (с выбором): {t0:.6f}")
        print(f"LU (общее): {t1:.6f}")

    df = pd.DataFrame({
        "Число правых частей k": ks,
        "Гаусс (с выбором)": gauss_times,
        "LU (разложение + решения)": lu_times
    })

    df.to_csv("experiment2.csv", index=False)

    # график
    plt.figure()
    plt.plot(ks, gauss_times, label="Гаусс (с выбором)")
    plt.plot(ks, lu_times, label="LU")

    plt.xlabel("k")
    plt.ylabel("Время (с)")
    plt.title("Зависимость времени от числа правых частей")
    plt.legend()
    plt.grid()

    plt.savefig("experiment2.png")
    plt.show()

    print("\nТаблица и график сохранены")


# =========================
# Эксперимент 3
# =========================
def experiment_hilbert():
    sizes = [5, 10, 15]

    results = []

    for n in sizes:
        print(f"\nМатрица Гильберта n = {n}")

        H = hilbert_matrix(n)
        x_true = np.ones(n)
        b = H @ x_true

        # Гаусс без выбора
        x_gauss = gauss(H, b)

        # Гаусс с выбором
        x_pivot = gauss_pivot(H, b)

        # LU
        L, U = lu_decomposition(H)
        x_lu = solve_lu(L, U, b)

        # Ошибки
        err_gauss = relative_error(x_gauss, x_true)
        err_pivot = relative_error(x_pivot, x_true)
        err_lu = relative_error(x_lu, x_true)

        # Невязки
        res_gauss = residual(H, x_gauss, b)
        res_pivot = residual(H, x_pivot, b)
        res_lu = residual(H, x_lu, b)

        results.append([
            n,
            err_gauss, err_pivot, err_lu,
            res_gauss, res_pivot, res_lu
        ])

        print("Ошибка Гаусс:", err_gauss)
        print("Ошибка Гаусс (с выбором):", err_pivot)
        print("Ошибка LU:", err_lu)

    df = pd.DataFrame(results, columns=[
        "Размер n",
        "Ошибка (Гаусс без выбора)",
        "Ошибка (Гаусс с выбором)",
        "Ошибка (LU)",
        "Невязка (Гаусс без выбора)",
        "Невязка (Гаусс с выбором)",
        "Невязка (LU)"
    ])

    df.to_csv("experiment3.csv", index=False)
    print("\nТаблица сохранена в experiment3.csv")
