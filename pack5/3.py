import random
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Функция генерации "почти нормальных" величин по ЦПТ
# ---------------------------------------------------------

def generate_normal_via_clt(n, N):
    """
    Генерирует N случайных величин, имеющих приблизительно
    стандартное нормальное распределение, через центральную
    предельную теорему.

    Для каждой η_j берётся своя последовательность
    ξ_1^(j), ..., ξ_n^(j), где ξ_i ~ U(0,1), независимы.

    Затем считается стандартизированная сумма:

        η_j = (ξ_1^(j) + ... + ξ_n^(j) - n*Eξ) / sqrt(n*Dξ)

    Для U(0,1):
        Eξ = 1/2,  Dξ = 1/12.
    """
    etas = []

    # параметры базового распределения (U(0,1))
    mu = 0.5            # Eξ
    var = 1.0 / 12.0    # Dξ

    for _ in range(N):
        s = 0.0
        for _ in range(n):
            xi = random.random()   # ξ ~ U(0,1)
            s += xi

        eta = (s - n * mu) / math.sqrt(n * var)
        etas.append(eta)

    return etas


# ---------------------------------------------------------
# Основная часть: выбор n и N, построение гистограммы
# ---------------------------------------------------------

def main():
    n = int(input("Введите n (длина суммы, например 5, 10, 50): "))
    N = int(input("Введите N (размер выборки, например 5000): "))

    random.seed(0)

    etas = generate_normal_via_clt(n, N)

    # Гистограмма η_1, ..., η_N
    plt.figure(figsize=(8, 5))
    plt.hist(etas, bins=50, density=True, alpha=0.6,
             edgecolor='black', label='Гистограмма η')

    # Теоретическая плотность стандартного нормального распределения
    x = np.linspace(-4, 4, 400)
    pdf = 1.0 / math.sqrt(2 * math.pi) * np.exp(-x**2 / 2)

    plt.plot(x, pdf, linewidth=2, label='Плотность N(0, 1)')

    plt.title(f"ЦПТ: n = {n}, N = {N}")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()