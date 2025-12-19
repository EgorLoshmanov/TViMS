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

    # -------------------------------
    # Задание №2:
    # Функция принимает список признаков (features) и на их основе строит МНК-оценки
    # параметров линейной регрессии, а также вычисляет оценки ошибок (остатки) и σ̂².
    # -------------------------------

    # 1. Формируем матрицу X и вектор y
    # y — наблюдения целевой переменной (price), X — матрица выбранных пользователем признаков
    y = df[target].to_numpy()                      # (n,)
    X = df[features].to_numpy()                    # (n, k)
    n = X.shape[0]                                 # n — число наблюдений
    k = X.shape[1]                                 # k — число признаков в модели (без свободного члена)

    # Добавляем столбец единиц для свободного члена beta_0
    # Это превращает X в дизайн-матрицу X_design размерности (n, k+1): [1, x1, x2, ..., xk]
    X_design = np.column_stack([np.ones(n), X])    # (n, k+1)

    # 2. Оценка коэффициентов по МНК: beta_hat = (X^T X)^(-1) X^T y
    # Это стандартная формула МНК-оценки параметров линейной регрессии
    XtX = X_design.T @ X_design                    # X^T X
    XtX_inv = np.linalg.inv(XtX)                   # (X^T X)^(-1)
    Xty = X_design.T @ y                           # X^T y
    beta_hat = XtX_inv @ Xty                       # (k+1,)

    # 3. Предсказания и остатки (оценки ошибок)
    # y_hat — прогнозы модели, residuals = y - y_hat — оценки eps_i (ошибок линейной регрессии)
    y_hat = X_design @ beta_hat                    # (n,)

    print("\nПервые 5 предсказанных цен:")
    print(y_hat[:5])

    residuals = y - y_hat

    # 4. Оценка дисперсии ошибок sigma^2:
    # RSS = Σ (epŝ_i)^2, а σ̂² = RSS / (n - (k+1)) — несмещённая оценка дисперсии ошибок
    rss = residuals @ residuals                    # сумма квадратов остатков
    sigma2_hat = rss / (n - k - 1)                 # делим на число степеней свободы: n - (k+1)

    # 5. Удобный словарь для коэффициентов
    # Имена параметров: intercept соответствует β0, далее коэффициенты при признаках из features
    param_names = ['intercept'] + features
    beta_hat_dict = {name: coef for name, coef in zip(param_names, beta_hat)}

    # Возвращаем
    # - оценки параметров (beta_hat_dict)
    # - оценки ошибок (residuals)
    # - оценку дисперсии ошибок (sigma2_hat)
    return beta_hat_dict, residuals, sigma2_hat


# Загружаем стандартный датасет (цены на дома и факторы)
df = pd.read_csv('pack6/data.csv')

# -------------------------------
# Задание №1: подготовка данных
# (1) удаление сильно коррелированных признаков,
# (2) one-hot кодирование категориальных переменных,
# (3) попытка улучшить связь признаков с price через преобразования
# -------------------------------

# -------------------------------
# 1. УДАЛЕНИЕ СИЛЬНО КОРРЕЛИРОВАННЫХ НЕЗАВИСИМЫХ ПРИЗНАКОВ
# -------------------------------
# Шаг 1 из задания №1:
# исключаем признаки с "слишком высокой" корреляцией |corr| > 0.9,
# чтобы уменьшить мультиколлинеарность в линейной модели
# Берём только независимые переменные (без целевой price)
features = df.drop(columns=['price'])

# Матрица корреляций только по числовым признакам
corr_matrix = features.corr(numeric_only=True)

# Верхний треугольник матрицы корреляций
# (чтобы не проверять пары дважды и не сравнивать признак с самим собой)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Список признаков, у которых есть хотя бы одна |corr| > 0.9 с другим признаком
# (эти признаки будут удалены из набора данных)
to_drop = [col for col in upper.columns if any(upper[col].abs() > 0.9)]

print("Удаляем сильно коррелированные признаки:", to_drop)

# Удаляем эти признаки из исходного датафрейма (целевую price не трогаем)
df_clean = df.drop(columns=to_drop)

# -------------------------------
# 2. ONE-HOT КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
# -------------------------------
# Шаг 2 из задания №1:
# преобразуем категориальные переменные в числовой формат через dummy/one-hot кодирование
cat_cols = ['airport', 'waterbody', 'bus_ter']

# drop_first=True удаляет одну категорию из каждой группы,
# чтобы избежать точной мультиколлинеарности (dummy trap) при наличии intercept
df_model = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

# -------------------------------
# 3. ТРАНСФОРМАЦИИ ПРИЗНАКОВ ДЛЯ УЛУЧШЕНИЯ КОРРЕЛЯЦИИ С price
# -------------------------------
# Шаг 3 из задания №1:
# для некоторых числовых признаков проверяем, улучшится ли связь с price
# после преобразований (log1p или sqrt). Если улучшилась — используем преобразованный признак.
target = 'price'

# Числовые признаки после one-hot
# (включая dummy-признаки, которые тоже числовые, но обычно бинарные 0/1)
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns
numeric_features = [col for col in numeric_cols if col != target]

for col in numeric_features:
    s = df_model[col]

    # Пропускаем бинарные и фиктивные признаки (0/1, максимум 2 уникальных значения)
    # Для них log/sqrt бессмысленны и обычно не меняют корреляцию адекватно
    if s.nunique() <= 2:
        continue

    # Исходная корреляция признака с price
    orig_corr = s.corr(df_model[target])

    # Храним лучший результат по модулю (важна сила связи, знак может быть + или -)
    best_corr = abs(orig_corr)
    best_series = s
    best_name = None

    # 3.1. Логарифмирование
    # log1p(x) = log(1+x) — безопаснее для малых значений; clip(lower=0) защищает от отрицательных
    s_log = np.log1p(s.clip(lower=0))
    log_corr = s_log.corr(df_model[target])
    if abs(log_corr) > best_corr:
        best_corr = abs(log_corr)
        best_series = s_log
        best_name = col + '_log'

    # 3.2. Квадратный корень
    # sqrt(x) может "сжать" большие значения и иногда делает зависимость ближе к линейной
    s_sqrt = np.sqrt(s.clip(lower=0))
    sqrt_corr = s_sqrt.corr(df_model[target])
    if abs(sqrt_corr) > best_corr:
        best_corr = abs(sqrt_corr)
        best_series = s_sqrt
        best_name = col + '_sqrt'

    # Если нашли преобразование с ЛУЧШЕЙ (по модулю) корреляцией — заменяем столбец
    # (оставляем преобразованную версию и удаляем исходную)
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
# Проверяем (диагностически), какие признаки теперь сильнее всего связаны с price
corr_with_price = (
    df_model.corr(numeric_only=True)['price']
    .sort_values(ascending=False)
)

print(corr_with_price)

# -------------------------------
# ПРИМЕР ИСПОЛЬЗОВАНИЯ fit_linear_regression
# -------------------------------
# Демонстрация задания №2:
# выбираем некоторый набор признаков (здесь — топ-5 по |corr| с price)
# и передаём список строк в fit_linear_regression(features=...)
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

# Строим МНК-модель и получаем то, что требуется по заданию №2:
# - оценки коэффициентов
# - оценки ошибок (остатки)
# - оценку σ̂²
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