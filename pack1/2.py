import random
import matplotlib.pyplot as plt

def main():
    x_number_of_points, y_current_value_pi = [], []
    total_points = 100
    points_on_circle = 0
    for i in range(1, total_points + 1):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        # Сторона квадрата = 1
        # Вписываем окружность в квадрат и получаем уравнение (x - 0.5)^2 + (y - 0.5)^2 = 0.25
        if (x - 0.5)**2 + (y - 0.5)**2 <= 0.25:
            points_on_circle += 1
        if i % 10 == 0:
            x_number_of_points.append(i)
            y_current_value_pi.append(4 * points_on_circle / i)

    print(4 * points_on_circle / total_points)
    plt.plot(x_number_of_points, y_current_value_pi)
    plt.xlabel("Кол-во точек")
    plt.ylabel("pi")
    plt.axhline(y=3.14, color='red', linestyle='--', linewidth=1.5)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()