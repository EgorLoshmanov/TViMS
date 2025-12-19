import numpy as np
import pandas as pd
from math import erf, sqrt


def normal_cdf(x: float) -> float:
    # Φ(x) — функция распределения стандартной нормали N(0,1)
    # Используется для вычисления p-value по статистике (z/t) при нормальной аппроксимации
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

    # y — целевая переменная (price), приводим к float на всякий случай
    y = df[target].astype(float).to_numpy()           # (n,)

    # X — матрица текущих признаков, приводим к float (после one-hot все должны стать числовыми)
    X = df[features].astype(float).to_numpy()         # (n, k)
    n, k = X.shape                                    # n — число наблюдений, k — число признаков

    # Добавляем столбец единиц для свободного члена (intercept = β0)
    X_design = np.column_stack([np.ones(n), X])       # (n, k+1)

    # Имена параметров в порядке коэффициентов в β̂
    param_names = ['intercept'] + features

    # --------------------
    # МНК-оценка β (OLS)
    # --------------------
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y

    # Решаем систему (XtX) * beta = Xty (устойчивее, чем явная инверсия)
    beta_hat = np.linalg.solve(XtX, Xty)              # (k+1,)

    # --------------------
    # Остатки и σ̂²
    # --------------------
    y_hat = X_design @ beta_hat
    residuals = y - y_hat
    rss = float(residuals @ residuals)

    dof = n - k - 1
    if dof <= 0:
        raise ValueError(f"Недостаточно наблюдений: dof = n-k-1 = {dof} (n={n}, k={k}).")

    sigma2_hat = rss / dof

    # --------------------
    # Дисперсии β̂ и стандартные ошибки
    # --------------------
    # Var(β̂) = σ̂² (X^T X)^(-1)
    # Для численной стабильности берём диагональ через solve, а не через inv целиком
    XtX_inv = np.linalg.inv(XtX)
    var_beta = sigma2_hat * np.diag(XtX_inv)
    se_beta = np.sqrt(var_beta)

    # --------------------
    # t-статистики и p-values
    # --------------------
    t_stats = beta_hat / se_beta

    p_values = {}
    for name, t in zip(param_names, t_stats):
        # Защита от nan/inf (например, если se_beta==0 или матрица плохо обусловлена)
        if not np.isfinite(t):
            p = 1.0
        else:
            p = 2 * (1 - normal_cdf(abs(float(t))))
            if not np.isfinite(p) or np.isnan(p):
                p = 1.0
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
                group_name = f
            groups.setdefault(group_name, []).append(f)
        return groups

    iteration = 1
    while True:
        print(f"\n=== Итерация {iteration} ===")
        print("Текущие признаки:", current_features)

        # Если признаков нет — останавливаемся
        if not current_features:
            print("Не осталось ни одного признака. Остановка.")
            break

        # 1) Оцениваем модель и p-value для всех коэффициентов
        param_names, beta_hat, p_values = fit_ols_and_pvalues(
            df, target, current_features
        )

        # 2) Строим группы (числовые + категориальные)
        groups = build_groups(current_features)

        # 3) Считаем group-p-value (для категории берём min p среди её dummy)
        group_pvalues = {}
        for group_name, cols in groups.items():
            p_list = []
            for col in cols:
                if col not in p_values:
                    continue
                p_list.append(p_values[col])
            if not p_list:
                continue
            gp = min(p_list)
            # Защита: если вдруг nan/inf — считаем группу незначимой (p=1)
            if not np.isfinite(gp) or np.isnan(gp):
                gp = 1.0
            group_pvalues[group_name] = gp

        # Если вообще не удалось посчитать ни одной группы — выходим
        if not group_pvalues:
            print("Не удалось вычислить group p-values. Остановка.")
            break

        print("group p-values:")
        for g, p in group_pvalues.items():
            print(f"  {g}: {p:.4g}")

        # 4) Проверка остановки
        max_group = max(group_pvalues, key=group_pvalues.get)
        max_p = float(group_pvalues[max_group])
        print(f"Максимальный group p-value = {max_p:.4g} (группа: {max_group})")

        if max_p < alpha:
            print("Все группы значимы (p < alpha). Останавливаемся.")
            break

        # 5) Удаляем группу с наибольшим p-value
        cols_to_remove = groups[max_group]
        print(f"Исключаем из модели группу '{max_group}' и её признаки:", cols_to_remove)

        current_features = [f for f in current_features if f not in cols_to_remove]
        iteration += 1

    return current_features


# ============================
# ПОДГОТОВКА ДАННЫХ ДЛЯ АЛГОРИТМА
# ============================

df = pd.read_csv("pack6/House-Prices.csv")
target = 'price'

# one-hot для категориальных переменных
cat_cols = ['airport', 'waterbody', 'bus_ter']
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 1) Убираем target и явный "утечка-признак" Sold (если он есть в датасете)
#    Sold нельзя использовать для предсказания price, т.к. он связан с ценой/продажей и ломает модель
excluded = {target, 'Sold'}

all_features = [col for col in df_model.columns if col not in excluded]

# 2) Убираем константные признаки (nunique <= 1), которые дают se=0 и nan в статистиках
all_features = [col for col in all_features if df_model[col].nunique(dropna=False) > 1]

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
