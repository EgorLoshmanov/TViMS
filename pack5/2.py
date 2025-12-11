import random
import math
import matplotlib.pyplot as plt

# ============================================================
# ФУНКЦИИ ВЫБОРОЧНЫХ ХАРАКТЕРИСТИК
# ============================================================

def U():
    """
    Генерирует число из равномерного распределения U(0,1).

    Returns
    -------
    float
        Случайное число в диапазоне [0, 1).
    """
    return random.random()


def bernoulli(p, size=1):
    """
    Генерация выборки из распределения Бернулли.

    Параметры
    ----------
    p : float
        Вероятность успеха (значения 1).
    size : int
        Размер выборки. Если size=1 — возвращается одно число.

    Returns
    -------
    int или list[int]
        0 или 1 (или список таких значений).

    Метод
    -----
    X = 1, если U < p; иначе 0.
    """
    if size == 1:
        return 1 if U() < p else 0
    return [1 if U() < p else 0 for _ in range(size)]


def binomial(n, p, size=1):
    """
    Генерация биномиальной случайной величины через сумму n испытаний Бернулли.

    Параметры
    ----------
    n : int
        Количество испытаний.
    p : float
        Вероятность успеха в одном испытании.
    size : int
        Размер выборки.

    Returns
    -------
    int или list[int]
        Число успехов от 0 до n.

    Метод
    -----
    X = сумма n независимых Бернулли(p).
    """
    def one():
        s = 0
        for _ in range(n):
            s += 1 if U() < p else 0
        return s
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def geometric(p, size=1):
    """
    Генерация геометрической случайной величины.

    Параметры
    ----------
    p : float
        Вероятность успеха.
    size : int
        Размер выборки.

    Returns
    -------
    int или list[int]
        Количество испытаний до первого успеха (1, 2, 3, ...).

    Метод
    -----
    Используется обратная функция распределения:
    X = 1 + floor( log(1-U) / log(1-p) )
    """
    def one():
        y = U()
        return 1 + int(math.log(1 - y) / math.log(1 - p))
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def poisson(lmbda, size=1):
    """
    Генерация распределения Пуассона по алгоритму Кнута.

    Параметры
    ----------
    lmbda : float
        Параметр λ (среднее значение).
    size : int
        Размер выборки.

    Returns
    -------
    int или list[int]
        Значение из распределения Пуассона.

    Метод
    -----
    Алгоритм Кнута: повторно умножаем U, пока произведение > exp(-λ).
    """
    def one():
        L = math.exp(-lmbda)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= U()
        return k - 1
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def uniform_ab(a, b, size=1):
    """
    Генерация равномерного распределения на отрезке [a, b].

    Параметры
    ----------
    a, b : float
        Границы интервала.
    size : int
        Размер выборки.

    Returns
    -------
    float или list[float]
        Значение из U(a, b).

    Метод
    -----
    X = a + (b - a) * U.
    """
    return [a + (b - a) * U() for _ in range(size)]


def exponential(alpha, size=1):
    """
    Генерация экспоненциального распределения с параметром α.

    Параметры
    ----------
    alpha : float
        Параметр интенсивности α.
    size : int
        Размер выборки.

    Returns
    -------
    float или list[float]
        Случайная величина >= 0.

    Метод
    -----
    Используется обратная CDF:
    X = -ln(1 - U) / α.
    """
    def one():
        return -math.log(1 - U()) / alpha
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def laplace(alpha, size=1):
    """
    Генерация распределения Лапласа (двусторонняя экспонента).

    Параметры
    ----------
    alpha : float
        Коэффициент затухания.
    size : int
        Размер выборки.

    Returns
    -------
    float или list[float]

    Метод
    -----
    Используется обратная функция распределения:
    если U <= 0.5:  X = (1/α) * ln(2U)
    иначе:          X = -(1/α) * ln(2(1-U))
    """
    def one():
        y = U()
        if y <= 0.5:
            return (1/alpha) * math.log(2 * y)
        else:
            return -(1/alpha) * math.log(2 * (1 - y))
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def normal(a, sigma, size=1):
    """
    Генерация нормального распределения N(a, σ²) по методу Бокса–Мюллера.

    Параметры
    ----------
    a : float
        Среднее значение.
    sigma : float
        Стандартное отклонение.
    size : int
        Размер выборки.

    Returns
    -------
    float или list[float]

    Метод
    -----
    Z = sqrt(-2 ln U1) * cos(2π U2)
    X = a + σZ
    """
    def one():
        u1 = U()
        u2 = U()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return a + sigma * z
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def cauchy(size=1):
    """
    Генерация стандартного распределения Коши.

    Параметры
    ----------
    size : int
        Размер выборки.

    Returns
    -------
    float или list[float]

    Метод
    -----
    Используется обратная CDF:
    X = tan(π(U - 1/2))
    """
    def one():
        return math.tan(math.pi * (U() - 0.5))
    if size == 1:
        return one()
    return [one() for _ in range(size)]


def custom_kernel(size=1):
    """
    Генерация распределения с плотностью f(t) = 2/t^3, t > 1.

    Параметры
    ----------
    size : int
        Размер выборки.

    Returns
    -------
    float или list[float]

    Метод
    -----
    Обратная функция распределения:
    G(t) = 1 - 1/t^2  =>  t = 1 / sqrt(1 - U)
    """
    def one():
        y = U()
        return 1 / math.sqrt(1 - y)
    if size == 1:
        return one()
    return [one() for _ in range(size)]

def sample_mean(sample):
    """
    Выборочное среднее:
        X̄ = (1/n) * sum_{i=1}^n X_i
    """
    n = len(sample)
    return sum(sample) / n


def sample_variance(sample):
    """
    Выборочная дисперсия (несмещённая):
        S^2 = 1/(n-1) * sum_{i=1}^n (X_i - X̄)^2
    """
    n = len(sample)
    x_bar = sample_mean(sample)
    return sum((x - x_bar) ** 2 for x in sample) / (n - 1)


def sample_skewness(sample):
    """
    Выборочный коэффициент асимметрии:

        γ̂_1 = [ (1/n) * sum (X_i - X̄)^3 ] / S^3

    где S^2 — выборочная дисперсия (несмещённая).
    """
    n = len(sample)
    x_bar = sample_mean(sample)
    s2 = sample_variance(sample)
    if s2 <= 0:
        return float('nan')
    s = math.sqrt(s2)

    m3 = sum((x - x_bar) ** 3 for x in sample) / n
    return m3 / (s ** 3)


def sample_kurtosis(sample):
    """
    Выборочный коэффициент эксцесса:

        γ̂_2 = [ (1/n) * sum (X_i - X̄)^4 ] / S^4  − 3

    где S^2 — выборочная дисперсия (несмещённая).
    """
    n = len(sample)
    x_bar = sample_mean(sample)
    s2 = sample_variance(sample)
    if s2 <= 0:
        return float('nan')

    m4 = sum((x - x_bar) ** 4 for x in sample) / n
    return m4 / (s2 ** 2) - 3


# ============================================================
# ДОПОЛНИТЕЛЬНЫЕ РАСПРЕДЕЛЕНИЯ: ГАММА И БЕТА
# ============================================================

def gamma_dist(k, theta, size=1):
    """
    Гамма-распределение с параметрами k (shape) и θ (scale).
    Используется random.gammavariate(k, theta).
    """
    def one():
        return random.gammavariate(k, theta)

    if size == 1:
        return one()
    return [one() for _ in range(size)]


def beta_dist(a, b, size=1):
    """
    Бета-распределение с параметрами a и b.
    Используется random.betavariate(a, b).
    """
    def one():
        return random.betavariate(a, b)

    if size == 1:
        return one()
    return [one() for _ in range(size)]


# ============================================================
# ИССЛЕДОВАНИЕ АСИММЕТРИИ И ЭКСЦЕССА
# ============================================================

random.seed(0)
size = 5000  # размер выборки для каждого эксперимента

experiments = [
    # --- распределения из задания 1 ---
    ("Bernoulli p=0.2",          lambda n: bernoulli(0.2, n)),
    ("Bernoulli p=0.5",          lambda n: bernoulli(0.5, n)),
    ("Bernoulli p=0.8",          lambda n: bernoulli(0.8, n)),

    ("Binomial n=10 p=0.3",      lambda n: binomial(10, 0.3, n)),
    ("Binomial n=10 p=0.5",      lambda n: binomial(10, 0.5, n)),

    ("Geometric p=0.3",          lambda n: geometric(0.3, n)),
    ("Geometric p=0.6",          lambda n: geometric(0.6, n)),

    ("Poisson λ=1",              lambda n: poisson(1, n)),
    ("Poisson λ=4",              lambda n: poisson(4, n)),
    ("Poisson λ=10",             lambda n: poisson(10, n)),

    ("Uniform [0,1]",            lambda n: uniform_ab(0, 1, n)),
    ("Uniform [2,5]",            lambda n: uniform_ab(2, 5, n)),

    ("Exponential α=1",          lambda n: exponential(1, n)),
    ("Exponential α=2",          lambda n: exponential(2, n)),

    ("Laplace α=1",              lambda n: laplace(1, n)),
    ("Laplace α=0.5",            lambda n: laplace(0.5, n)),

    ("Normal μ=0 σ=1",           lambda n: normal(0, 1, n)),
    ("Normal μ=0 σ=2",           lambda n: normal(0, 2, n)),

    ("Cauchy",                   lambda n: cauchy(n)),

    ("Custom f(t)=2/t^3 t>1",    lambda n: custom_kernel(n)),

    # --- гамма и бета-распределения ---
    ("Gamma k=2 θ=1",            lambda n: gamma_dist(2, 1, n)),
    ("Gamma k=5 θ=1",            lambda n: gamma_dist(5, 1, n)),
    ("Gamma k=2 θ=2",            lambda n: gamma_dist(2, 2, n)),

    ("Beta a=0.5 b=0.5",         lambda n: beta_dist(0.5, 0.5, n)),
    ("Beta a=2 b=5",             lambda n: beta_dist(2, 5, n)),
    ("Beta a=5 b=2",             lambda n: beta_dist(5, 2, n)),
]

# Табличный вывод в консоль
print(f"{'Распределение':35s}  {'Асимметрия':>12s}  {'Эксцесс':>12s}")
print("-" * 65)

for name, gen in experiments:
    sample = gen(size)
    g1 = sample_skewness(sample)
    g2 = sample_kurtosis(sample)

    print(f"{name:35s}  {g1:12.3f}  {g2:12.3f}")

    # Гистограмма
    plt.figure(figsize=(7, 4))
    plt.hist(sample, bins=50, density=True, edgecolor='black', alpha=0.6)
    plt.title(f"{name}\nАсимметрия = {g1:.2f}, Эксцесс = {g2:.2f}")
    plt.xlabel("Значение")
    plt.ylabel("Плотность / частота")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.show(block=False)        # открыть окно не блокируя код
    plt.waitforbuttonpress()     # ждать нажатия
    plt.close()                  # закрыть окноа