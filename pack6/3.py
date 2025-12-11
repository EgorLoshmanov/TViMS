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

DATA_FILE = "House-Prices.csv"
WATER_GROUPS = ['None', 'River', 'Lake', 'Lake and River']
PLOT_X_RANGE = (0.001, 0.999, 400)

# Априорные предположения для каждой группы waterbody
USER_PRIORS = {
    'None':           {'mean': 0.3, 'std': 0.15},
    'River':          {'mean': 0.5, 'std': 0.15},
    'Lake':           {'mean': 0.6, 'std': 0.15},
    'Lake and River': {'mean': 0.7, 'std': 0.15},
}

# Порядок объединения групп для последовательного обновления
UPDATE_ORDER = ['None', 'River', 'Lake', 'Lake and River']


# ---------------------------
# КЛАССЫ ДАННЫХ
# ---------------------------

@dataclass
class BetaDistribution:
    """Представление Beta-распределения с параметрами alpha и beta."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Математическое ожидание."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Дисперсия."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        """Стандартное отклонение."""
        return sqrt(self.variance)

    def pdf(self, x: np.ndarray) -> np.ndarray:
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
        if not (0 < mean < 1):
            raise ValueError(f"mean должен быть в (0,1), получено: {mean}")

        variance = std ** 2
        if variance <= 0:
            raise ValueError(f"std должен быть > 0, получено: {std}")

        max_variance = mean * (1 - mean)
        if variance >= max_variance:
            raise ValueError(
                f"Слишком большое std={std:.4f} для mean={mean:.4f}: "
                f"должно быть std^2 < mean*(1-mean) ≈ {max_variance:.4f}"
            )

        t = mean * (1 - mean) / variance - 1
        alpha = mean * t
        beta = (1 - mean) * t

        return cls(alpha=alpha, beta=beta)


@dataclass
class GroupAnalysisResult:
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
    """
    Загрузка и подготовка данных о продажах недвижимости.

    Args:
        filepath: Путь к CSV-файлу

    Returns:
        DataFrame с обработанными данными
    """
    df = pd.read_csv(filepath)
    df['water_group'] = df['waterbody'].fillna('None')

    print("Уникальные значения water_group:", df['water_group'].unique())

    return df


def create_prior_distributions(priors_config: Dict) -> Dict[str, BetaDistribution]:
    """
    Создание априорных распределений для всех групп.

    Args:
        priors_config: Словарь с конфигурацией априорных распределений

    Returns:
        Словарь с BetaDistribution для каждой группы
    """
    distributions = {}

    for group, params in priors_config.items():
        distribution = BetaDistribution.from_mean_std(
            mean=params['mean'],
            std=params['std']
        )
        distributions[group] = distribution
        print(f"Группа {group}: априор Beta(α={distribution.alpha:.2f}, β={distribution.beta:.2f})")

    return distributions


def analyze_group(df: pd.DataFrame, group: str, prior: BetaDistribution) -> GroupAnalysisResult:
    """
    Анализ одной группы: вычисление апостериорного распределения.

    Args:
        df: DataFrame с данными
        group: Название группы для анализа
        prior: Априорное распределение

    Returns:
        Результат анализа группы
    """
    group_data = df[df['water_group'] == group]
    n_observations = len(group_data)

    if n_observations == 0:
        raise ValueError(f"Группа {group}: нет данных")

    n_sold = group_data['Sold'].sum()
    n_not_sold = n_observations - n_sold

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
    """
    Последовательное байесовское обновление при объединении групп.

    Args:
        df: DataFrame с данными
        groups_order: Порядок обработки групп
        initial_prior: Начальное априорное распределение

    Returns:
        Список пар (метка, распределение) для каждого шага
    """
    current_distribution = initial_prior
    distributions = [('prior_' + groups_order[0], current_distribution)]

    for group in groups_order:
        group_data = df[df['water_group'] == group]
        n_observations = len(group_data)

        if n_observations == 0:
            continue

        n_sold = group_data['Sold'].sum()
        n_not_sold = n_observations - n_sold

        current_distribution = current_distribution.update(
            successes=n_sold,
            failures=n_not_sold
        )

        label = f'after_{group}'
        distributions.append((label, current_distribution))

    return distributions


# ---------------------------
# ФУНКЦИИ ВЫВОДА РЕЗУЛЬТАТОВ
# ---------------------------

def print_group_results(results: List[GroupAnalysisResult]) -> None:
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

    print("\nРезультаты по группам (априор → апостериор):")
    print(df_results[['group', 'n_obs', 'sold', 'prior_mean', 'prior_std',
                      'post_mean', 'post_std']].sort_values('post_mean', ascending=False))

    print("\nСортировка по апостериорной вероятности продажи (post_mean):")
    print(df_results[['group', 'post_mean', 'post_std']].sort_values('post_mean', ascending=False))


def print_sequential_updates(distributions: List[Tuple[str, BetaDistribution]]) -> None:
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
    """
    Визуализация априорных и апостериорных распределений для каждой группы.

    Args:
        results: Список результатов анализа
        x_range: Диапазон значений x (start, stop, num_points)
    """
    x = np.linspace(*x_range)

    plt.figure(figsize=(10, 6))

    for result in results:
        prior_pdf = result.prior.pdf(x)
        posterior_pdf = result.posterior.pdf(x)

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
    """
    Визуализация последовательных байесовских обновлений.

    Args:
        distributions: Список пар (метка, распределение)
        x_range: Диапазон значений x (start, stop, num_points)
    """
    x = np.linspace(*x_range)

    plt.figure(figsize=(10, 6))

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
    """Основная функция для выполнения полного анализа."""
    # Загрузка данных
    df = load_and_prepare_data(DATA_FILE)

    # Создание априорных распределений
    prior_distributions = create_prior_distributions(USER_PRIORS)

    # Анализ каждой группы
    results = []
    for group in WATER_GROUPS:
        try:
            result = analyze_group(df, group, prior_distributions[group])
            results.append(result)
        except ValueError as e:
            print(f"Предупреждение: {e}")
            continue

    # Вывод результатов
    print_group_results(results)

    # Последовательные обновления
    sequential_distributions = perform_sequential_updates(
        df=df,
        groups_order=UPDATE_ORDER,
        initial_prior=prior_distributions['None']
    )
    print_sequential_updates(sequential_distributions)

    # Визуализация
    plot_group_distributions(results, PLOT_X_RANGE)
    plot_sequential_updates(sequential_distributions, PLOT_X_RANGE)


if __name__ == '__main__':
    main()
