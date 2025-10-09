import numpy as np

def main():
    X = np.random.exponential(scale=1.0, size=10000)
    n = len(X)
    number_of_samples = int(input("Введите число выборок для бутстрапа: "))

    bootstrap_samples = [np.random.choice(X, size=n, replace=True) for _ in range(number_of_samples)]

    # ===== MAD =====
    MAD_values = []
    for sample in bootstrap_samples:
        median = np.median(sample)
        MAD = np.median(np.abs(sample - median))
        MAD_values.append(MAD)
    MAD_values = np.asarray(MAD_values)

    MAD_lower, MAD_upper = np.percentile(MAD_values, [2.5, 97.5])

    # 95% ДИ (кратчайший интервал, HDI)
    sorted_vals = np.sort(MAD_values)
    k = int(np.floor(0.95 * len(sorted_vals)))  # размер окна
    # ширины всех возможных "окон" длиной k
    widths = sorted_vals[k:] - sorted_vals[:-k]
    i_min = np.argmin(widths)
    MAD_hdi_lower, MAD_hdi_upper = sorted_vals[i_min], sorted_vals[i_min + k]

    print('\n' + '-' * 50)
    print(f"Число бутстрап-выборок: {number_of_samples}")
    print(f"Точечная оценка MAD (по исходной выборке): "
        f"{np.median(np.abs(X - np.median(X))):.5f}")
    print(f"95% ДИ (percentile equal-tailed): [{MAD_lower:.5f}, {MAD_upper:.5f}]")
    print(f"95% ДИ (кратчайший, HDI):         [{MAD_hdi_lower:.5f}, {MAD_hdi_upper:.5f}]")

    print("\nПервые 10 MAD из бутстрап-выборок:")
    print(np.round(MAD_values[:10], 5))

    # ===== Выборочная дисперсия (ddof=1) =====
    var_values = []
    for sample in bootstrap_samples:
        var = np.var(sample, ddof=1)
        var_values.append(var)
    var_values = np.asarray(var_values)

    var_lower, var_upper = np.percentile(var_values, [2.5, 97.5])

    # 95% ДИ (кратчайший интервал, HDI)
    sorted_vals = np.sort(var_values)
    k = int(np.floor(0.95 * len(sorted_vals)))  # размер окна
    # ширины всех возможных "окон" длиной k
    widths = sorted_vals[k:] - sorted_vals[:-k]
    i_min = np.argmin(widths)
    var_hdi_lower, var_hdi_upper = sorted_vals[i_min], sorted_vals[i_min + k]

    print('\n' + '-' * 50)
    print(f"Число бутстрап-выборок: {number_of_samples}")
    print(f"Точечная оценка var (по исходной выборке): "
        f"{np.var(X, ddof=1):.5f}")
    print(f"95% ДИ (percentile equal-tailed): [{var_lower:.5f}, {var_upper:.5f}]")
    print(f"95% ДИ (кратчайший, HDI):         [{var_hdi_lower:.5f}, {var_hdi_upper:.5f}]")

    print("\nПервые 10 var из бутстрап выборок:")
    print(np.round(var_values[:10], 5))

if __name__ == "__main__":
    main()