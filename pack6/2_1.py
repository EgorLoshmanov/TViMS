import numpy as np
import pandas as pd
from math import erf, erfc, sqrt


def normal_cdf(x: float) -> float:
    # Φ(x) — функция распределения стандартной нормали N(0,1)
    # Используется в статистических тестах для перевода z-статистики в вероятность
    """Стандартная нормальная CDF Φ(x)"""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_two_sided_pvalue_from_z(z: float) -> float:
    # Двусторонний p-value для проверки H0: β_s = 0 против H1: β_s ≠ 0
    # При H0 стандартизованный коэффициент (z) считается примерно ~ N(0,1)
    # Тогда p = 2 * P(Z >= |z|) = 2*(1 - Φ(|z|))
    # Используем erfc, потому что 1-Φ(|z|) при больших |z| может терять точность
    """
    Двусторонний p-value для z-статистики N(0,1), численно устойчиво.
    p = 2*(1-Φ(|z|)) = erfc(|z|/sqrt(2))
    """
    return erfc(abs(z) / sqrt(2.0))


def test_coef_significance(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    coef_name: str
):
    """
    Проводит тест значимости коэффициента β_s в линейной регрессии.

    Возвращает словарь с:
      - beta_hat      : оценка коэффициента β̂_s
      - var_beta_hat  : оценка дисперсии D̂(β̂_s)
      - se_beta_hat   : стандартная ошибка SE(β̂_s)
      - beta_st       : стандартизованный коэффициент (z-статистика)
      - p_value       : двусторонний p-value (устойчиво)
      - sigma2_hat    : оценка дисперсии ошибок σ̂^2
      - dof           : степени свободы (n - k - 1)
    """

    # ---------------------------------------------------------
    # Задание №3 (проверка значимости коэффициента β_s):
    # 1) Оценить дисперсию β̂_s, подставив σ̂² (из остатков) вместо σ²:
    #    D̂(β̂) = σ̂² * (X^T X)^(-1), значит D̂(β̂_s) = σ̂² * [(X^T X)^(-1)]_{ss}
    # 2) Найти стандартизацию при H0: β_s = 0:
    #    z = (β̂_s - 0) / SE(β̂_s)
    # 3) Предположить, что z ~ N(0,1) примерно, и посчитать двусторонний p-value:
    #    p = 2 * (1 - Φ(|z|))
    # ---------------------------------------------------------

    # --- 1) Формируем X и y, добавляем столбец единиц ---
    # y — целевая переменная (например, price)
    y = df[target].to_numpy(dtype=float)          # (n,)

    # X — матрица признаков, указанных пользователем (список строк features)
    X = df[features].to_numpy(dtype=float)        # (n, k)
    n, k = X.shape                                # n — наблюдения, k — число признаков

    # Добавляем столбец единиц для свободного члена β0 (intercept)
    X_design = np.column_stack([np.ones(n), X])   # (n, k+1)

    # Список имён коэффициентов в том же порядке, как в β-векторе
    param_names = ['intercept'] + features

    # --- 2) Оценка β по МНК ---
    # МНК-оценка β̂ решает (X^T X) β̂ = X^T y
    XtX = X_design.T @ X_design                   # (k+1, k+1)
    Xty = X_design.T @ y                          # (k+1,)

    # Решаем систему линейных уравнений (обычно устойчивее, чем инверсия)
    beta_hat_vec = np.linalg.solve(XtX, Xty)      # (k+1,)

    # --- 3) Остатки и оценка σ^2 ---
    # Предсказания: ŷ = X β̂
    y_hat = X_design @ beta_hat_vec

    # Остатки: epŝ = y - ŷ (оценки ошибок регрессии)
    residuals = y - y_hat

    # RSS = Σ epŝ_i^2 — сумма квадратов остатков
    rss = float(residuals @ residuals)

    # Степени свободы: n - (k+1), т.к. оцениваем (k+1) параметров (с intercept)
    dof = n - k - 1
    if dof <= 0:
        # Если dof <= 0, σ̂² не определена (деление на 0 или отрицательное)
        raise ValueError(f"Недостаточно наблюдений: dof = n-k-1 = {dof} (n={n}, k={k}).")

    # Оценка дисперсии ошибок σ̂² (используется дальше вместо неизвестной σ²)
    sigma2_hat = rss / dof

    # --- 4) Дисперсия и стандартизация нужного коэффициента ---
    # Проверяем, что coef_name существует среди доступных коэффициентов
    if coef_name not in param_names:
        raise ValueError(f"Коэффициент {coef_name} не найден. Доступно: {param_names}")

    # Индекс нужного коэффициента β_s в векторе β̂
    idx = param_names.index(coef_name)

    # (X^T X)^(-1) — нужно для дисперсии оценок коэффициентов
    XtX_inv = np.linalg.inv(XtX)

    # Оценка дисперсии коэффициента β̂_s:
    # D̂(β̂_s) = σ̂² * [(X^T X)^(-1)]_{ss}
    var_beta_hat = sigma2_hat * XtX_inv[idx, idx]

    # Стандартная ошибка: SE(β̂_s) = sqrt(D̂(β̂_s))
    se_beta_hat = float(np.sqrt(var_beta_hat))

    # Точечная оценка коэффициента β̂_s
    beta_hat = float(beta_hat_vec[idx])

    # Если SE=0, z = β̂/SE не определён (деление на 0)
    if se_beta_hat == 0.0:
        # В пределе: если β̂ != 0, то z -> ∞ и p-value -> 0
        beta_st = float('inf') if beta_hat != 0 else 0.0
        p_value = 0.0 if beta_hat != 0 else 1.0
    else:
        # Стандартизация при H0: β_s = 0
        # z = (β̂_s - 0) / SE(β̂_s)
        beta_st = beta_hat / se_beta_hat

        # --- 5) p-value (устойчиво): через erfc, а не через 1 - Φ ---
        # Двусторонний p-value: p = 2*(1 - Φ(|z|))
        p_value = normal_two_sided_pvalue_from_z(beta_st)

    # Возвращаем все величины, которые прямо перечислены в условии задания №3
    return {
        "beta_hat": beta_hat,                 # β̂_s
        "var_beta_hat": float(var_beta_hat),  # D̂(β̂_s)
        "se_beta_hat": se_beta_hat,           # SE(β̂_s)
        "beta_st": float(beta_st),            # z-статистика
        "p_value": float(p_value),            # p-value
        "sigma2_hat": float(sigma2_hat),      # σ̂²
        "dof": int(dof),                      # n-k-1
    }


# ------------------ пример использования ------------------
# Загружаем данные (исходный датасет)
df = pd.read_csv("pack6/House-Prices.csv")

# Задаём целевую переменную и набор признаков для модели линейной регрессии
target = "price"
features = ["room_num", "poor_prop", "resid_area"]

# Запускаем тест значимости выбранного коэффициента (например, при room_num)
result = test_coef_significance(df, target, features, coef_name="room_num")

# Печатаем результаты теста значимости
print("Оценка коэффициента β̂_s:", result["beta_hat"])
print("Оценка дисперсии D̂(β̂_s):", result["var_beta_hat"])
print("Стандартная ошибка SE(β̂_s):", result["se_beta_hat"])
print("Стандартизованный коэффициент (z):", result["beta_st"])

# p-value часто бывает очень маленьким, поэтому выводим и в scientific, и в фиксированном формате
print("p-value (scientific):", f"{result['p_value']:.30e}")
print("p-value:", format(result["p_value"], ".20"))
print("dof:", result["dof"])
