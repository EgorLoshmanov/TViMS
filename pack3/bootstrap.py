import numpy as np   

def main():
    X = np.random.exponential(scale=1.0, size=10000)
    n = len(X)  

    # Запрос у пользователя количества бутстрап-выборок (например, 1000, 5000 и т.д.)
    number_of_samples = int(input("Введите число выборок для бутстрапа: "))

    # Формирование списка бутстрап-выборок:
    # из исходной выборки X случайно выбираются n элементов с возвращением (replace=True)
    # таких выборок создаётся number_of_samples штук
    bootstrap_samples = [np.random.choice(X, size=n, replace=True) for _ in range(number_of_samples)]

    # ===== MAD (Median Absolute Deviation — медианное абсолютное отклонение) =====
    MAD_values = []  # сюда будут сохраняться значения MAD для каждой бутстрап-выборки

    for sample in bootstrap_samples:
        median = np.median(sample)  # медиана текущей бутстрап-выборки
        MAD = np.median(np.abs(sample - median))  # медиана абсолютных отклонений от медианы
        MAD_values.append(MAD)  # добавляем результат в список

    MAD_values = np.asarray(MAD_values) 

    # Двусторонние 95% доверительные границы (percentile method)
    MAD_lower, MAD_upper = np.percentile(MAD_values, [5, 95,])

    # ===== Вычисление 95% HDI (Highest Density Interval — кратчайший интервал) =====
    # Сначала сортируем значения MAD
    sorted_vals = np.sort(MAD_values)
    # Определяем размер окна, соответствующего 95% выборки
    k = int(np.floor(0.90 * len(sorted_vals)))
    # Для всех возможных "окон" длиной k считаем ширину (разность между концами)
    widths = sorted_vals[k:] - sorted_vals[:-k]
    # Находим индекс окна с минимальной шириной - оно и будет HDI
    i_min = np.argmin(widths)
    MAD_hdi_lower, MAD_hdi_upper = sorted_vals[i_min], sorted_vals[i_min + k]

    # === Вывод результатов по MAD ===
    print('\n' + '-' * 50)
    print(f"Число бутстрап-выборок: {number_of_samples}")
    print(f"Точечная оценка MAD (по исходной выборке): "
        f"{np.median(np.abs(X - np.median(X))):.5f}")  # вычисляем MAD исходной выборки
    print(f"95% ДИ (percentile equal-tailed): [{MAD_lower:.5f}, {MAD_upper:.5f}]")
    print(f"95% ДИ (кратчайший, HDI):         [{MAD_hdi_lower:.5f}, {MAD_hdi_upper:.5f}]")

    # Выводим первые 10 значений MAD для проверки
    print("\nПервые 10 MAD из бутстрап-выборок:")
    print(np.round(MAD_values[:10], 5))

    # ===== Выборочная дисперсия =====
    var_values = []  # список для дисперсий по каждой бутстрап-выборке

    for sample in bootstrap_samples:
        var = np.var(sample, ddof=1)  # выборочная дисперсия текущей выборки
        var_values.append(var)
    var_values = np.asarray(var_values)

    # 95% доверительный интервал по процентилям
    var_lower, var_upper = np.percentile(var_values, [5, 95])

    # ===== Вычисление HDI для дисперсии =====
    sorted_vals = np.sort(var_values)
    k = int(np.floor(0.90 * len(sorted_vals)))  # длина окна для 95%
    widths = sorted_vals[k:] - sorted_vals[:-k]  # ширины всех возможных окон
    i_min = np.argmin(widths)  # индекс минимальной ширины окна
    var_hdi_lower, var_hdi_upper = sorted_vals[i_min], sorted_vals[i_min + k]

    # === Вывод результатов по дисперсии ===
    print('\n' + '-' * 50)
    print(f"Число бутстрап-выборок: {number_of_samples}")
    print(f"Точечная оценка var (по исходной выборке): "
        f"{np.var(X, ddof=1):.5f}")  # дисперсия исходной выборки
    print(f"95% ДИ (percentile equal-tailed): [{var_lower:.5f}, {var_upper:.5f}]")
    print(f"95% ДИ (кратчайший, HDI):         [{var_hdi_lower:.5f}, {var_hdi_upper:.5f}]")

    print("\nПервые 10 var из бутстрап выборок:")
    print(np.round(var_values[:10], 5))

if __name__ == "__main__":
    main()