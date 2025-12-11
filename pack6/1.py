import numpy as np
import pandas as pd


def fit_linear_regression(df: pd.DataFrame, target: str, features: list[str]):
    """
    df       : DataFrame с данными
    target   : имя целевой переменной (строка), например 'price'
    features : список строк с именами признаков, например ['room_num', 'poor_prop_log']

    Возвращает:
      - beta_hat_dict : словарь {имя_параметра: значение_оценки}
      - residuals     : вектор остатков (оценки ошибок eps_i)
      - sigma2_hat    : оценка дисперсии ошибок sigma^2
    """

    # 1. Формируем матрицу X и вектор y
    y = df[target].to_numpy()                      # (n,)
    X = df[features].to_numpy()                    # (n, k)
    n = X.shape[0]
    k = X.shape[1]

    # Добавляем столбец единиц для свободного члена beta_0
    X_design = np.column_stack([np.ones(n), X])    # (n, k+1)

    # 2. Оценка коэффициентов по МНК: beta_hat = (X^T X)^(-1) X^T y
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_design.T @ y
    beta_hat = XtX_inv @ Xty                      # (k+1,)

    # 3. Предсказания и остатки (оценки ошибок)
    y_hat = X_design @ beta_hat                   # (n,)
    residuals = y - y_hat                         # \hat eps_i

    # 4. Оценка дисперсии ошибок sigma^2:
    #    \hat sigma^2 = (1 / (n - k - 1)) * sum(residuals^2)
    rss = residuals @ residuals                   # сумма квадратов остатков
    sigma2_hat = rss / (n - k - 1)

    # 5. Сделаем удобный словарь для коэффициентов
    param_names = ['intercept'] + features
    beta_hat_dict = {name: coef for name, coef in zip(param_names, beta_hat)}

    return beta_hat_dict, residuals, sigma2_hat


df = pd.read_csv('pack6/data.csv')

# -------------------------------
# 1. УДАЛЕНИЕ СИЛЬНО КОРРЕЛИРОВАННЫХ НЕЗАВИСИМЫХ ПРИЗНАКОВ
# -------------------------------
# Берём только независимые переменные (без целевой price)
features = df.drop(columns=['price'])

# Матрица корреляций только по числовым признакам
corr_matrix = features.corr(numeric_only=True)

# Верхний треугольник матрицы корреляций
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Список признаков, у которых есть хотя бы одна |corr| > 0.9 с другим признаком
to_drop = [col for col in upper.columns if any(upper[col].abs() > 0.9)]

print("Удаляем сильно коррелированные признаки:", to_drop)

# Удаляем эти признаки из исходного датафрейма (целевую price не трогаем)
df_clean = df.drop(columns=to_drop)

# -------------------------------
# 2. ONE-HOT КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
# -------------------------------
cat_cols = ['airport', 'waterbody', 'bus_ter']

df_model = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

# -------------------------------
# 3. ТРАНСФОРМАЦИИ ПРИЗНАКОВ ДЛЯ УЛУЧШЕНИЯ КОРРЕЛЯЦИИ С price
# -------------------------------
target = 'price'

# Числовые признаки после one-hot
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns
numeric_features = [col for col in numeric_cols if col != target]

for col in numeric_features:
    s = df_model[col]

    # Пропускаем бинарные и фиктивные признаки (0/1, максимум 2 уникальных значения)
    if s.nunique() <= 2:
        continue

    orig_corr = s.corr(df_model[target])
    best_corr = abs(orig_corr)
    best_series = s
    best_name = None

    # 3.1. Логарифмирование (log1p с обрезкой снизу для безопасности)
    s_log = np.log1p(s.clip(lower=0))
    log_corr = s_log.corr(df_model[target])
    if abs(log_corr) > best_corr:
        best_corr = abs(log_corr)
        best_series = s_log
        best_name = col + '_log'

    # 3.2. Квадратный корень
    s_sqrt = np.sqrt(s.clip(lower=0))
    sqrt_corr = s_sqrt.corr(df_model[target])
    if abs(sqrt_corr) > best_corr:
        best_corr = abs(sqrt_corr)
        best_series = s_sqrt
        best_name = col + '_sqrt'

    # Если нашли преобразование с ЛУЧШЕЙ (по модулю) корреляцией — заменяем столбец
    if best_name is not None:
        print(f"Для признака {col}: исходная corr={orig_corr:.3f}, "
              f"лучшая после преобразования={best_corr:.3f}, используем {best_name}")
        # добавляем новый столбец
        df_model[best_name] = best_series
        # удаляем исходный
        df_model.drop(columns=[col], inplace=True)

# -------------------------------
# ИТОГ: смотрим корреляции с целевой переменной
# -------------------------------
corr_with_price = (
    df_model.corr(numeric_only=True)['price']
    .sort_values(ascending=False)
)

print(corr_with_price)

# -------------------------------
# ПРИМЕР ИСПОЛЬЗОВАНИЯ fit_linear_regression
# -------------------------------

# Берём, например, топ-5 признаков с наибольшей по модулю корреляцией с price (кроме самой price)
top_features = (
    corr_with_price.drop('price')      # убираем целевую
                  .abs()               # берём модуль корреляции
                  .sort_values(ascending=False)
                  .head(5)             # топ-5
                  .index
                  .tolist()
)

print("\nПризнаки, используемые в модели линейной регрессии:")
print(top_features)

beta_hat_dict, residuals, sigma2_hat = fit_linear_regression(
    df=df_model,
    target=target,
    features=top_features
)

print("\nОценки коэффициентов линейной регрессии:")
for name, coef in beta_hat_dict.items():
    print(f"{name}: {coef:.4f}")

print(f"\nОценка дисперсии ошибок sigma^2: {sigma2_hat:.4f}")

print("\nПервые 5 остатков (оценки ошибок eps_i):")
print(residuals[:5])
