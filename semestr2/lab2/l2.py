import matplotlib.pyplot as plt
import numpy as np

def trapez(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    return 0.0

def main():
    print("вариант - Загруженность дорог")
    try:
        print("\nпараметры трапециевидной функции (a, b, c, d):")
        a = float(input("a (начало подъема): "))
        b = float(input("b (начало полки): "))
        c = float(input("c (конец полки): "))
        d = float(input("d (конец спада): "))

        raw_points = input("\nВведите четкие значения через пробел (например, 10 30 55 80): ")
        points = [float(x) for x in raw_points.split()]

        print("\nРезультаты расчета:")
        print(f"{'Объект (x)':<15} | {'mu(x) (Исходное)':<20} | {'mu_comp(x) (Дополнение)':<20}")
        print("-" * 65)

        for x in points:
            mu = trapez(x, a, b, c, d)
            mu_complement = 1 - mu
            print(f"{x:<15.2f} | {mu:<20.4f} | {mu_complement:<20.4f}")


        x_min = min(a, min(points)) - 5
        x_max = max(d, max(points)) + 5
        x_vals = np.linspace(x_min, x_max, 500)

        mu_vals = [trapez(x, a, b, c, d) for x in x_vals]
        mu_comp_vals = [1 - mu for mu in mu_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


        ax1.plot(x_vals, mu_vals, 'b-', label='μ(x)')
        ax1.scatter(points, [trapez(x, a, b, c, d) for x in points], color='red', zorder=5, label='Введённые точки')
        ax1.set_title('Функция принадлежности "Загруженность дорог"')
        ax1.set_xlabel('Интенсивность движения (x)')
        ax1.set_ylabel('μ(x)')
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        ax2.plot(x_vals, mu_comp_vals, 'r-', label='1 - μ(x)')
        ax2.scatter(points, [1 - trapez(x, a, b, c, d) for x in points], color='blue', zorder=5, label='Введённые точки')
        ax2.set_title('Дополнение функции принадлежности')
        ax2.set_xlabel('Интенсивность движения (x)')
        ax2.set_ylabel('1 - μ(x)')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ValueError:
        print("только числовые значения.")
    except ZeroDivisionError:
        print("Ошибка: Параметры a, b или c, d не должны быть равны для наклонных участков.")

if __name__ == "__main__":
    main()
