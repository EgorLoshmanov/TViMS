import numpy as np
import pandas as pd
from math import erf, sqrt


def normal_cdf(x: float) -> float:
    """Стандартная нормальная CDF Φ(x) без scipy."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def test_coef_significance(df: pd.DataFrame,
                           target: str,
                           features: list[str],
                           coef_name: str):
    """
    Проводит тест значимости коэффициента β_s в линейной регрессии.

    df        : DataFrame с данными
    target    : имя целевой переменной, например 'price'
    features  : список признаков, которые включаем в модель
    coef_name : имя коэффициента, значимость которого тестируем.
                Может быть 'intercept' или один из features.

    Возвращает словарь с:
      - beta_hat      : оценка коэффициента β̂_s
      - var_beta_hat  : оценка дисперсии D̂(β̂_s)
      - beta_st       : стандартизованный коэффициент β̂_s^(st)
      - p_value       : двусторонний p-value
    """

    # --- 1. Формируем X и y, добавляем столбец единиц ---
    y = df[target].to_numpy()                # (n,)
    X = df[features].to_numpy()              # (n, k)
    n = X.shape[0]
    k = X.shape[1]

    X_design = np.column_stack([np.ones(n), X])  # (n, k+1)
    param_names = ['intercept'] + features

    # --- 2. Оценка β по формуле МНК ---
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_design.T @ y
    beta_hat_vec = XtX_inv @ Xty             # (k+1,)

    # --- 3. Остатки и оценка σ^2 ---
    y_hat = X_design @ beta_hat_vec
    residuals = y - y_hat
    rss = residuals @ residuals
    sigma2_hat = rss / (n - k - 1)           # несмещённая оценка σ^2

    # --- 4. Дисперсия и стандартизация нужного коэффициента ---
    if coef_name not in param_names:
        raise ValueError(f"Коэффициент {coef_name} не найден в модели.")

    idx = param_names.index(coef_name)

    var_beta_hat = sigma2_hat * XtX_inv[idx, idx]   # D̂(β̂_s)
    se_beta_hat = np.sqrt(var_beta_hat)             # стандартная ошибка
    beta_hat = beta_hat_vec[idx]

    beta_st = beta_hat / se_beta_hat                # β̂_s^(st)

    # --- 5. p-value для двустороннего теста H0: β_s = 0 ---
    p_value = 2 * (1 - normal_cdf(abs(beta_st)))    # две стороны

    return {
        "beta_hat": beta_hat,
        "var_beta_hat": var_beta_hat,
        "beta_st": beta_st,
        "p_value": p_value,
        "sigma2_hat": sigma2_hat
    }

df = pd.read_csv("House-Prices.csv")

target = "price"
features = ["room_num", "poor_prop", "resid_area"]  # любые твои признаки

result = test_coef_significance(df, target, features, coef_name="room_num")

print("Оценка коэффициента β̂_s:", result["beta_hat"])
print("Оценка дисперсии D̂(β̂_s):", result["var_beta_hat"])
print("Стандартизованный коэффициент β̂_s^(st):", result["beta_st"])
print("p-value:", result["p_value"])
