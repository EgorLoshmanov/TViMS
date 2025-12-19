"""
Байесовский анализ вероятности продажи недвижимости с использованием Beta-распределения.
Анализируется влияние близости к водоёмам на вероятность продажи.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import lgamma, sqrt


# ---------------------------
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ---------------------------

# Путь к исходному датасету (используем исходный набор данных о домах)
DATA_FILE = "pack6/House-Prices.csv"

# Группы по признаку waterbody 
WATER_GROUPS = ['None', 'River', 'Lake', 'Lake and River']

# Диапазон p для графиков Beta-плотности: (start, stop, num_points)
# (не берём 0 и 1, чтобы избежать log(0) в pdf)
PLOT_X_RANGE = (0.001, 0.999, 400)

# Априорные предположения пользователя:
# для каждой группы задаём априорное матожидание p и "неуверенность" (априорное std)
# Согласно заданию №4 считаем, что априор для p — Beta(α, β)
USER_PRIORS = {
    'None':           {'mean': 0.3, 'std': 0.15},
    'River':          {'mean': 0.5, 'std': 0.15},
    'Lake':           {'mean': 0.6, 'std': 0.15},
    'Lake and River': {'mean': 0.7, 'std': 0.15},
}

# Порядок объединения групп для демонстрации последовательных обновлений (из задания №4)
UPDATE_ORDER = ['None', 'River', 'Lake', 'Lake and River']


# ---------------------------
# КЛАССЫ ДАННЫХ
# ---------------------------

@dataclass
class BetaDistribution:
    # Обёртка над Beta(α, β), чтобы удобно:
    # - хранить параметры,
    # - считать mean/std,
    # - делать байесовское обновление (α+=успехи, β+=неудачи),
    # - строить график плотности
    """Представление Beta-распределения с параметрами alpha и beta."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        # Матожидание Beta: E[p] = α / (α+β)
        """Математическое ожидание."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        # Дисперсия Beta: αβ / ((α+β)^2 (α+β+1))
        """Дисперсия."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        # Стандартное отклонение = sqrt(variance)
        """Стандартное отклонение."""
        return sqrt(self.variance)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        # Плотность Beta(p | α,β) в точках x (0<x<1)
        # Считаем в логарифмах для численной устойчивости:
        # log pdf = (α-1)log x + (β-1)log(1-x) - log B(α,β),
        # где log B(α,β) = lgamma(α)+lgamma(β)-lgamma(α+β)
        """
        Численно стабильная плотность вероятности.

        Args:
            x: Точки для вычисления плотности (0 < x < 1)

        Returns:
            Значения плотности вероятности
        """
        x = np.asarray(x)
        log_beta_func = lgamma(self.alpha) + lgamma(self.beta) - lgamma(self.alpha + self.beta)
        log_pdf = (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - log_beta_func
        return np.exp(log_pdf)

    def update(self, successes: int, failures: int) -> 'BetaDistribution':
        # Байесовское обновление для Бернулли/Биномиальной модели:
        # prior: Beta(α,β)
        # данные: successes = число продаж (1), failures = число непродаж (0)
        # posterior: Beta(α+successes, β+failures)
        # Это ровно то, что требует задание №4 "используя априор и данные по группе"
        """
        Байесовское обновление на основе новых данных.

        Args:
            successes: Количество успехов
            failures: Количество неудач

        Returns:
            Новое апостериорное распределение
        """
        return BetaDistribution(
            alpha=self.alpha + successes,
            beta=self.beta + failures
        )

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> 'BetaDistribution':
        # Перевод пользовательских параметров (mean, std) -> (α, β)
        # Для Beta:
        # mean = α/(α+β)
        # var  = αβ / ((α+β)^2 (α+β+1))
        # Выражаем t = α+β:
        # var = mean(1-mean)/(t+1)  =>  t = mean(1-mean)/var - 1
        # затем α = mean * t, β = (1-mean) * t
        """
        Создание Beta-распределения из среднего и стандартного отклонения.

        Args:
            mean: Среднее значение (должно быть в интервале (0, 1))
            std: Стандартное отклонение

        Returns:
            BetaDistribution с соответствующими параметрами

        Raises:
            ValueError: Если параметры некорректны
        """
        # Проверяем корректность среднего (вероятность должна быть между 0 и 1)
        if not (0 < mean < 1):
            raise ValueError(f"mean должен быть в (0,1), получено: {mean}")

        # Переводим std в дисперсию
        variance = std ** 2
        if variance <= 0:
            raise ValueError(f"std должен быть > 0, получено: {std}")

        # Для Beta максимально возможная дисперсия при данном mean равна mean*(1-mean)
        # Если variance >= mean*(1-mean), то подходящего Beta(α,β) не существует (α,β стали бы <=0)
        max_variance = mean * (1 - mean)
        if variance >= max_variance:
            raise ValueError(
                f"Слишком большое std={std:.4f} для mean={mean:.4f}: "
                f"должно быть std^2 < mean*(1-mean) ≈ {max_variance:.4f}"
            )

        # Вспомогательная величина t = α+β
        t = mean * (1 - mean) / variance - 1

        # Восстанавливаем α и β по mean и t
        alpha = mean * t
        beta = (1 - mean) * t

        return cls(alpha=alpha, beta=beta)


@dataclass
class GroupAnalysisResult:
    # Структура "всё по группе" — удобно хранить и печатать итог:
    # - сколько наблюдений
    # - сколько продано/не продано
    # - априор и апостериор
    """Результат анализа для одной группы."""
    group: str
    n_observations: int
    n_sold: int
    n_not_sold: int
    prior: BetaDistribution
    posterior: BetaDistribution


# ---------------------------
# ФУНКЦИИ ЗАГРУЗКИ И ОБРАБОТКИ ДАННЫХ
# ---------------------------

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    # Загрузка исходного датасета и формирование групп по waterbody, как требует задание №4:
    # значения waterbody могут быть NaN => считаем это группой 'None'
    """
    Загрузка и подготовка данных о продажах недвижимости.

    Args:
        filepath: Путь к CSV-файлу

    Returns:
        DataFrame с обработанными данными
    """
    df = pd.read_csv(filepath)

    # Создаём явный столбец-группу (вместо NaN пишем 'None')
    df['water_group'] = df['waterbody'].fillna('None')

    # Диагностический вывод: какие группы реально встретились в данных
    print("Уникальные значения water_group:", df['water_group'].unique())

    return df


def create_prior_distributions(priors_config: Dict) -> Dict[str, BetaDistribution]:
    # Создаём априор Beta(α,β) для каждой группы из пользовательских mean/std
    # (это прямо первая часть задания №4: "пользователь задаёт априорное матожидание и std")
    """
    Создание априорных распределений для всех групп.

    Args:
        priors_config: Словарь с конфигурацией априорных распределений

    Returns:
        Словарь с BetaDistribution для каждой группы
    """
    distributions = {}

    for group, params in priors_config.items():
        # Перевод (mean,std) -> (α,β)
        distribution = BetaDistribution.from_mean_std(
            mean=params['mean'],
            std=params['std']
        )
        distributions[group] = distribution

        # Печать получившихся параметров априора (чтобы было видно, какой prior используется)
        print(f"Группа {group}: априор Beta(α={distribution.alpha:.2f}, β={distribution.beta:.2f})")

    return distributions


def analyze_group(df: pd.DataFrame, group: str, prior: BetaDistribution) -> GroupAnalysisResult:
    # Для конкретной группы:
    # 1) выбираем строки этой группы
    # 2) считаем число "успехов" (Sold=1) и "неудач" (Sold=0)
    # 3) делаем байесовское обновление Beta prior -> posterior
    # 4) возвращаем структуру результата (для печати и графиков)
    """
    Анализ одной группы: вычисление апостериорного распределения.

    Args:
        df: DataFrame с данными
        group: Название группы для анализа
        prior: Априорное распределение

    Returns:
        Результат анализа группы
    """
    # Подтаблица по одной группе water_group
    group_data = df[df['water_group'] == group]
    n_observations = len(group_data)

    # Если данных нет — анализ невозможен
    if n_observations == 0:
        raise ValueError(f"Группа {group}: нет данных")

    # Число продаж (успехов); предполагается, что Sold — бинарный столбец 0/1
    n_sold = group_data['Sold'].sum()

    # Число непродаж (неудач)
    n_not_sold = n_observations - n_sold

    # Апостериор по группе: Beta(α+n_sold, β+n_not_sold)
    posterior = prior.update(successes=n_sold, failures=n_not_sold)

    return GroupAnalysisResult(
        group=group,
        n_observations=n_observations,
        n_sold=n_sold,
        n_not_sold=n_not_sold,
        prior=prior,
        posterior=posterior
    )


def perform_sequential_updates(
    df: pd.DataFrame,
    groups_order: List[str],
    initial_prior: BetaDistribution
) -> List[Tuple[str, BetaDistribution]]:
    # Вторая часть задания №4:
    # показать последовательные байесовские обновления при "объединении" групп:
    # берём prior (по группе None), обновляем по данным None -> получаем posterior_1,
    # затем используем posterior_1 как новый prior и обновляем по данным следующей группы, и т.д.
    """
    Последовательное байесовское обновление при объединении групп.

    Args:
        df: DataFrame с данными
        groups_order: Порядок обработки групп
        initial_prior: Начальное априорное распределение

    Returns:
        Список пар (метка, распределение) для каждого шага
    """
    # Текущее распределение p, которое будем по шагам уточнять данными
    current_distribution = initial_prior

    # Сохраняем начальный prior (чтобы на графике и в выводе был "старт")
    distributions = [('prior_' + groups_order[0], current_distribution)]

    # Проходим группы в заданном порядке и "добавляем" их данные в обновление
    for group in groups_order:
        group_data = df[df['water_group'] == group]
        n_observations = len(group_data)

        # Если группа пуста — пропускаем
        if n_observations == 0:
            continue

        # Считаем успехи/неудачи в этой группе
        n_sold = group_data['Sold'].sum()
        n_not_sold = n_observations - n_sold

        # Последовательное обновление:
        # prior_{step} -> posterior_{step} с добавлением данных текущей группы
        current_distribution = current_distribution.update(
            successes=n_sold,
            failures=n_not_sold
        )

        # Метка шага для печати/легенды графика
        label = f'after_{group}'
        distributions.append((label, current_distribution))

    return distributions


# ---------------------------
# ФУНКЦИИ ВЫВОДА РЕЗУЛЬТАТОВ
# ---------------------------

def print_group_results(results: List[GroupAnalysisResult]) -> None:
    # Выводим для каждой группы:
    # - сколько наблюдений
    # - сколько продано
    # - prior mean/std (ввод пользователя)
    # - posterior mean/std (после учёта данных)
    # Это соответствует требованию задания №4: "выведите матожидание и std апостериора"
    # и "сравните группы: где выше/ниже вероятность и как изменилась неопределённость"
    """
    Вывод результатов анализа групп в табличном виде.

    Args:
        results: Список результатов анализа
    """
    data = []
    for r in results:
        data.append({
            'group': r.group,
            'n_obs': r.n_observations,
            'sold': r.n_sold,
            'prior_mean': r.prior.mean,
            'prior_std': r.prior.std,
            'post_mean': r.posterior.mean,
            'post_std': r.posterior.std,
        })

    df_results = pd.DataFrame(data)

    # Таблица "априор -> апостериор" по всем группам
    print("\nРезультаты по группам (априор → апостериор):")
    print(df_results[['group', 'n_obs', 'sold', 'prior_mean', 'prior_std',
                      'post_mean', 'post_std']].sort_values('post_mean', ascending=False))

    # Дополнительно сортируем по post_mean, чтобы прямо видеть сравнение групп
    print("\nСортировка по апостериорной вероятности продажи (post_mean):")
    print(df_results[['group', 'post_mean', 'post_std']].sort_values('post_mean', ascending=False))


def print_sequential_updates(distributions: List[Tuple[str, BetaDistribution]]) -> None:
    # Печать каждого шага последовательного обновления:
    # показываем параметры Beta(α,β), а также mean/std, чтобы было видно "уточнение" распределения
    """
    Вывод информации о последовательных обновлениях.

    Args:
        distributions: Список пар (метка, распределение)
    """
    print("\nПоследовательные обновления (объединение групп):")
    for label, dist in distributions:
        print(f"{label}: Beta(α={dist.alpha:.2f}, β={dist.beta:.2f}), "
              f"mean={dist.mean:.4f}, std={dist.std:.4f}")


# ---------------------------
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ---------------------------

def plot_group_distributions(results: List[GroupAnalysisResult], x_range: Tuple[float, float, int]) -> None:
    # Визуализация первой части задания №4:
    # рисуем для каждой группы две кривые:
    # - prior Beta(α,β) (штрих)
    # - posterior Beta(α+n_sold, β+n_not_sold) (сплошная)
    # чтобы было видно, как данные "сдвигают" и "сужают" распределение вероятности продажи p
    """
    Визуализация априорных и апостериорных распределений для каждой группы.

    Args:
        results: Список результатов анализа
        x_range: Диапазон значений x (start, stop, num_points)
    """
    # Сетка значений p (вероятности) для построения плотности
    x = np.linspace(*x_range)

    plt.figure(figsize=(10, 6))

    for result in results:
        # Плотность prior и posterior для текущей группы
        prior_pdf = result.prior.pdf(x)
        posterior_pdf = result.posterior.pdf(x)

        # Рисуем prior пунктиром, posterior — сплошной линией
        plt.plot(x, prior_pdf, linestyle='--', label=f'{result.group} prior', alpha=0.7)
        plt.plot(x, posterior_pdf, label=f'{result.group} posterior', linewidth=2)

    plt.title('Априорные и апостериорные распределения для групп waterbody')
    plt.xlabel('p (вероятность продажи)')
    plt.ylabel('Плотность вероятности')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sequential_updates(
    distributions: List[Tuple[str, BetaDistribution]],
    x_range: Tuple[float, float, int]
) -> None:
    # Визуализация второй части задания №4:
    # рисуем исходный prior и все распределения после каждого последовательного обновления
    # (чтобы видно было постепенное "уточнение" по мере объединения групп)
    """
    Визуализация последовательных байесовских обновлений.

    Args:
        distributions: Список пар (метка, распределение)
        x_range: Диапазон значений x (start, stop, num_points)
    """
    # Сетка значений p
    x = np.linspace(*x_range)

    plt.figure(figsize=(10, 6))

    # Каждая кривая — плотность Beta на текущем шаге обновления
    for label, dist in distributions:
        pdf = dist.pdf(x)
        plt.plot(x, pdf, label=label, linewidth=2)

    plt.title('Последовательные байесовские обновления при объединении групп')
    plt.xlabel('p (вероятность продажи)')
    plt.ylabel('Плотность вероятности')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# ---------------------------

def main() -> None:
    # Точка входа: запускаем все шаги задания №4 последовательно
    """Основная функция для выполнения полного анализа."""
    # Загрузка данных и формирование столбца water_group
    df = load_and_prepare_data(DATA_FILE)

    # Создание априорных Beta-распределений по пользовательским mean/std для каждой группы
    prior_distributions = create_prior_distributions(USER_PRIORS)

    # Анализ каждой группы отдельно:
    # получаем апостериор Beta для каждой группы и сохраняем результаты
    results = []
    for group in WATER_GROUPS:
        try:
            result = analyze_group(df, group, prior_distributions[group])
            results.append(result)
        except ValueError as e:
            # Если в группе нет данных — не падаем, а предупреждаем и идём дальше
            print(f"Предупреждение: {e}")
            continue

    # Выводим сравнение групп: post_mean/post_std (и prior для сравнения неопределённости)
    print_group_results(results)

    # Последовательные обновления при объединении групп в заданном порядке
    sequential_distributions = perform_sequential_updates(
        df=df,
        groups_order=UPDATE_ORDER,
        initial_prior=prior_distributions['None']  # стартуем с априора пользователя для группы None
    )
    print_sequential_updates(sequential_distributions)

    # Графики: (1) prior/posterior по каждой группе, (2) последовательные обновления
    plot_group_distributions(results, PLOT_X_RANGE)
    plot_sequential_updates(sequential_distributions, PLOT_X_RANGE)


# Стандартный Python-шаблон: запускаем main() только если файл запущен как программа
if __name__ == '__main__':
    main()