import matplotlib.pyplot as plt
import time

def pseudorandom_number_generator(x0, a, b, M):
    while True:
        x0 = (a * x0 + b) % M
        yield x0

def main():
    SEED = time.time_ns()

    x0, a, b, M = SEED, 6364136223846793005, 1442695040888963407, 2**64
    # x0, a, b, M = 7, 17, 10, 19

    generator = pseudorandom_number_generator(x0, a, b, M)

    x_number_of_points, y_current_value_pi = [], []
    total_points = 10**6
    points_on_circle = 0
    for i in range(1, total_points+1):
        x = next(generator) / M
        next(generator)
        y = next(generator) / M
        # Сторона квадрата = 1
        # Вписываем окружность в квадрат и получаем уравнение (x - 0.5)^2 + (y - 0.5)^2 = 0.25
        if (x - 0.5)**2 + (y - 0.5)**2 <= 0.25:
            points_on_circle += 1
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