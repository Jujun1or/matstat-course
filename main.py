import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

np.random.seed(42)

# папки
Path("results").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)


def tukey_outliers_fraction(data):
    """Доля выбросов по правилу Тьюки"""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    mask = (data < low) | (data > high)
    return np.sum(mask) / len(data)


# распределения
dists = [
    ("Нормальное", lambda n: np.random.normal(0, 1, n), (-4, 4), "C0"),
    ("Коши", lambda n: np.random.standard_cauchy(n), (-15, 15), "C1"),
    ("Лапласа", lambda n: np.random.laplace(0, 1/np.sqrt(2), n), (-4, 4), "C2"),
    ("Пуассона", lambda n: np.random.poisson(5, n), (-2, 12), "C3"),
    ("Равномерное", lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n), (-2, 2), "C4")
]

sizes = [20, 100]
R = 1000

csv_path = "results/outliers.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Распределение", "n", "Доля выбросов"])


for name, gen, xlim, color in dists:
    for n in sizes:

        fractions = []

        for _ in range(R):
            sample = gen(n)
            fractions.append(tukey_outliers_fraction(sample))

        avg = np.mean(fractions)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([name, n, f"{avg:.4f}"])

        # боксплот
        plt.figure(figsize=(5, 3))

        plt.boxplot(
            sample,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor=color),
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(
                marker='o',
                markerfacecolor='red',
                markeredgecolor='red',
                markersize=4,
                linestyle='none'
            )
        )

        plt.title(f"{name}, n={n}")
        plt.xlabel("Значения")
        plt.xlim(xlim)

        plt.tight_layout()
        plt.savefig(f"images/{name}_n{n}.png", dpi=150)
        plt.close()
