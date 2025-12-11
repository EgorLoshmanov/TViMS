import random
import math
import matplotlib.pyplot as plt

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


random.seed(0)
size = 5000

distributions = {
    "Bernoulli p=0.3": bernoulli(0.3, size),
    "Binomial n=10 p=0.5": binomial(10, 0.5, size),
    "Geometric p=0.3": geometric(0.3, size),
    "Poisson λ=3": poisson(3, size),
    "Uniform [2,5]": uniform_ab(2, 5, size),
    "Exponential α=1": exponential(1, size),
    "Laplace α=1": laplace(1, size),
    "Normal a=0 σ=1": normal(0, 1, size),
    "Cauchy": cauchy(size),
    "Custom kernel f(t)=1/t^3 t>1": custom_kernel(size)
}

# Строим графики
for name, sample in distributions.items():
    plt.figure(figsize=(7,4))
    plt.hist(sample, bins=50, density=True, edgecolor='black')
    plt.title(name)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
