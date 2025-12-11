import numpy as np
import pandas as pd
from math import erf, sqrt


def normal_cdf(x: float) -> float:
    """Стандартная нормальная CDF Φ(x) без scipy."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def fit_ols_and_pvalues(df: pd.DataFrame, target: str, features: list[str]):
    """
    Строит линейную регрессию по признакам features и
    возвращает:
      - param_names : ['intercept'] + features
      - beta_hat    : оценки коэффициентов
      - p_values    : словарь {имя_параметра: p-value} (включая intercept)
    """
    # ЯВНО приводим к float, чтобы не было dtype=object
    y = df[target].astype(float).to_numpy()           # (n,)
    X = df[features].astype(float).to_numpy()         # (n, k)
    n, k = X.shape

    X_design = np.column_stack([np.ones(n), X])       # (n, k+1)
    param_names = ['intercept'] + features

    # МНК-оценка β
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_design.T @ y
    beta_hat = XtX_inv @ Xty

    # Остатки и σ^2
    y_hat = X_design @ beta_hat
    residuals = y - y_hat
    rss = residuals @ residuals
    sigma2_hat = rss / (n - k - 1)

    # Дисперсии β̂ и стандартные ошибки
    var_beta = sigma2_hat * np.diag(XtX_inv)
    se_beta = np.sqrt(var_beta)

    # t-статистики и p-values
    t_stats = beta_hat / se_beta
    p_values = {}
    for name, t in zip(param_names, t_stats):
        p = 2 * (1 - normal_cdf(abs(t)))
        p_values[name] = p

    return param_names, beta_hat, p_values

def backward_elimination(df: pd.DataFrame,
                         target: str,
                         all_features: list[str],
                         cat_cols: list[str],
                         alpha: float = 0.05):
    """
    Реализует backward elimination с группировкой дамми-колонок
    по исходным категориальным переменным.
    """

    current_features = all_features.copy()

    def build_groups(features_list):
        """Группируем признаки: числовые сами по себе, дамми по исходному cat."""
        groups = {}
        for f in features_list:
            group_name = None
            for cat in cat_cols:
                prefix = cat + "_"
                if f.startswith(prefix):
                    group_name = cat
                    break
            if group_name is None:
                group_name = f  # отдельный числовой признак

            groups.setdefault(group_name, []).append(f)
        return groups

    iteration = 1
    while True:
        print(f"\n=== Итерация {iteration} ===")
        print("Текущие признаки:", current_features)

        # 1. Оцениваем модель и p-value для всех коэффициентов
        param_names, beta_hat, p_values = fit_ols_and_pvalues(
            df, target, current_features
        )

        # 2. Строим группы (числовые + категориальные)
        groups = build_groups(current_features)

        # 3. Считаем group-p-value
        group_pvalues = {}
        for group_name, cols in groups.items():
            p_list = []
            for col in cols:
                if col not in p_values:
                    continue
                p_list.append(p_values[col])
            if not p_list:
                continue
            group_pvalues[group_name] = min(p_list)

        print("group p-values:")
        for g, p in group_pvalues.items():
            print(f"  {g}: {p:.4g}")

        # 4. Проверка остановки
        max_group = max(group_pvalues, key=group_pvalues.get)
        max_p = group_pvalues[max_group]
        print(f"Максимальный group p-value = {max_p:.4g} (группа: {max_group})")

        if max_p < alpha:
            print("Все группы значимы (p < alpha). Останавливаемся.")
            break

        # 5. Удаляем группу с наибольшим p-value
        cols_to_remove = groups[max_group]
        print(f"Исключаем из модели группу '{max_group}' и её признаки:",
              cols_to_remove)

        current_features = [f for f in current_features if f not in cols_to_remove]

        if not current_features:
            print("Не осталось ни одного признака. Остановка.")
            break

        iteration += 1

    return current_features


# ============================
# ПОДГОТОВКА ДАННЫХ ДЛЯ АЛГОРИТМА
# ============================

df = pd.read_csv("House-Prices.csv")
target = 'price'

# one-hot для категориальных переменных
cat_cols = ['airport', 'waterbody', 'bus_ter']
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# все признаки, кроме целевой
all_features = [col for col in df_model.columns if col != target]

# запуск backward elimination
final_features = backward_elimination(
    df=df_model,
    target=target,
    all_features=all_features,
    cat_cols=cat_cols,
    alpha=0.05
)

print("\nИТОГОВЫЕ ПРИЗНАКИ ПОСЛЕ BACKWARD ELIMINATION:")
print(final_features)
