import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


def simulate_bernoulli_poisson(lmbda: float, n: int, N: int, seed: int | None = None):
    """
    Моделирует схему Бернулли на [0, 1] N раз для заданных λ и n.
    + Строит нормированные гистограммы для количества успехов и расстояний между ними.

    Параметры
    ----------
    lmbda : float
        Параметр λ (интенсивность).
    n : int
        Количество подотрезков разбиения [0, 1].
    N : int
        Количество запусков алгоритма (итераций).
    seed : int | None
        Начальное зерно ГСЧ (для воспроизводимости). Можно не задавать.

    Возвращает
    ----------
    counts : np.ndarray shape = (N,)
        Количество успехов в каждой из N итераций.
    distances : np.ndarray shape = (M,)
        Расстояния между соседними успехами (по всем итерациям сразу),
        измеренные между центрами ячеек.
        M — случайное число, зависит от реализации испытаний.
    """
    rng = np.random.default_rng(seed)

    # Вероятность успеха в одной ячейке
    p = min(lmbda / n, 1.0)

    # Массив для количества успехов
    counts = np.empty(N, dtype=int)

    # Список для расстояний между успехами
    distances_list: list[float] = []

    # Центры всех ячеек [0,1]: (i + 0.5) / n
    cell_centers = (np.arange(n) + 0.5) / n

    for k in range(N):
        # Генерируем n независимых испытаний Бернулли
        successes = rng.random(n) < p  # булев массив длины n

        # Считаем число успехов
        counts[k] = successes.sum()

        # Индексы успешных ячеек
        success_indices = np.flatnonzero(successes)

        # Если успехов хотя бы два — считаем расстояния между соседними
        if success_indices.size >= 2:
            centers = cell_centers[success_indices]
            d = np.diff(centers)  # расстояния между соседними центрами
            distances_list.extend(d.tolist())

    distances = np.array(distances_list, dtype=float)

    # -----------------------------------------------------------
    # Построение нормированных гистограмм
    # -----------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Гистограмма количества успехов
    axs[0].hist(counts, bins=1 + int(np.log2(N)), density=True, color="skyblue", edgecolor="black")
    axs[0].set_xlabel("Количество успехов за итерацию")
    axs[0].set_ylabel("Плотность вероятности")
    axs[0].set_title("Гистограмма количества успехов (counts)")

    # 2. Гистограмма расстояний между успехами
    if len(distances) > 0:
        axs[1].hist(distances, bins=1 + int(np.log2(len(distances))), density=True,
                    color="salmon", edgecolor="black")
        axs[1].set_xlabel("Расстояние между соседними успехами")
        axs[1].set_ylabel("Плотность вероятности")
        axs[1].set_title("Гистограмма расстояний (distances)")

        # Теоретическая экспоненциальная кривая для сравнения
        x = np.linspace(0, distances.max(), 200)
        axs[1].plot(x, lmbda * np.exp(-lmbda * x), "r-", lw=2,
                    label=f"Exp(λ={lmbda})")
        axs[1].legend()

    plt.suptitle(f"λ={lmbda}, n={n}, N={N}")
    plt.tight_layout()
    plt.show()

    return counts, distances


def simulate_poisson_process(lmbda: float, N: int, seed: int | None = None):
    """
    Моделирует Пуассоновский процесс на отрезке [0, 1] N раз
    и строит гистограмму расстояний между успехами.

    Параметры
    ----------
    lmbda : float
        Параметр λ (интенсивность процесса, среднее число событий на [0,1]).
    N : int
        Количество итераций моделирования.
    seed : int | None
        Зерно генератора случайных чисел для воспроизводимости (необязательно).

    Возвращает
    ----------
    distances : np.ndarray
        Массив расстояний между соседними успехами, собранный со всех итераций.
        Его длина случайна, зависит от количества успехов во всех реализациях.
    """
    rng = np.random.default_rng(seed)
    distances_all: list[float] = []

    for _ in range(N):
        # 1. Случайное количество событий по закону Пуассона
        k = rng.poisson(lmbda)

        if k >= 2:
            # 2. Генерация k равномерных точек на [0,1]
            points = rng.random(k)
            points.sort()

            # 3. Расстояния между соседними точками
            diffs = np.diff(points)
            distances_all.extend(diffs.tolist())

    distances = np.array(distances_all)

    # --- Построение нормированной гистограммы ---
    if len(distances) > 0:
        m = 1 + int(np.log2(len(distances)))  # число столбцов по Стёрджессу
        plt.hist(distances, bins=m, density=True,
                 color="skyblue", edgecolor="black", alpha=0.7, label="эмпирическая плотность")

        # Теоретическая экспоненциальная кривая λ e^{-λt}
        x = np.linspace(0, distances.max(), 300)
        plt.plot(x, lmbda * np.exp(-lmbda * x), "r-", lw=2, label=f"Exp(λ={lmbda})")

        plt.title(f"Распределение расстояний между успехами (λ={lmbda}, N={N})")
        plt.xlabel("Расстояние между событиями")
        plt.ylabel("Плотность вероятности")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return distances


def simulate_poisson_counts(lmbda: float, N: int, seed: int | None = None):
    """
    Моделирует процесс Пуассона на [0, 1] N раз,
    используя экспоненциальные интервалы между событиями.
    + Строит гистограмму количества успехов (count) и сравнивает с теоретическим распределением Пуассона.

    Параметры
    ----------
    lmbda : float
        Параметр λ (интенсивность процесса).
    N : int
        Количество реализаций (итераций моделирования).
    seed : int | None
        Зерно генератора случайных чисел (для воспроизводимости, необязательно).

    Возвращает
    ----------
    counts : np.ndarray shape = (N,)
        Количество событий (успехов) в каждой итерации.
    """
    rng = np.random.default_rng(seed)
    counts = np.empty(N, dtype=int)

    for i in range(N):
        t = 0.0
        count = 0

        # генерируем события, пока не выйдем за границу [0, 1]
        while True:
            dt = rng.exponential(1.0 / lmbda)  # интервал ~ Exp(λ)
            t += dt
            if t > 1.0:
                break
            count += 1

        counts[i] = count

    # --- Построение гистограммы ---
    m = 1 + int(np.log2(N))  # правило Стёрджесса (необязательно, можно просто range)
    max_k = counts.max()

    plt.hist(counts, bins=np.arange(-0.5, max_k + 1.5, 1),
             density=True, color="skyblue", edgecolor="black", alpha=0.7,
             label="Эмпирическая частота")

    # Теоретическая кривая Пуассона
    x = np.arange(0, max_k + 1)
    plt.plot(x, poisson.pmf(x, lmbda), "r-", lw=2, label=f"Pois(λ={lmbda})")

    plt.title(f"Распределение количества событий N(1) (λ={lmbda}, N={N})")
    plt.xlabel("Количество успехов за итерацию")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return counts


def main():

    # --- Пример для 1 задания --- 
    lmbda = 10
    n = 24
    N = 500

    counts, distances = simulate_bernoulli_poisson(lmbda, n, N, seed=42)

    print("Кол-во успехов в первых 10 итерациях:", counts[:10])
    print("Всего расстояний между успехами:", len(distances))
    print("Первые 10 расстояний:", distances[:10])


    # --- Пример для 2 задания --- 
    distances = simulate_poisson_process(lmbda=10, N=500, seed=42)

    print("Количество всех расстояний:", len(distances))
    print("Первые 10 расстояний:", distances[:10])


    # --- Пример для 4 задания ---
    lmbda = 10
    N = 500

    counts = simulate_poisson_counts(lmbda, N, seed=42)

    print("Среднее количество успехов:", counts.mean())
    print("Теоретическое значение λ:", lmbda)
    print("Первые 10 значений:", counts[:10])

if __name__ == "__main__":
    main()