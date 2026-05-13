import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform
from pathlib import Path

np.random.seed(42)

# =========================
# ПАПКА ДЛЯ ГРАФИКОВ
# =========================

outdir = Path("images")
outdir.mkdir(exist_ok=True)

# =========================
# ШИРИНА ОКНА KDE
# =========================

def silverman_bandwidth(sample):
    n = len(sample)
    sigma = np.std(sample, ddof=1)

    return 1.06 * sigma * n ** (-1 / 5)


def robust_bandwidth(sample):
    n = len(sample)

    sigma = np.std(sample, ddof=1)

    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)

    iqr = q3 - q1

    scale = min(sigma, iqr / 1.34)

    return 0.9 * scale * n ** (-1 / 5)


# =========================
# ГАУССОВО ЯДРО
# =========================

def gaussian_kernel_density(x, sample, h):

    density = np.zeros_like(x)

    for value in sample:
        density += np.exp(-0.5 * ((x - value) / h) ** 2)

    density /= (len(sample) * h * np.sqrt(2 * np.pi))

    return density


# =========================
# РАСПРЕДЕЛЕНИЯ
# =========================

distributions = [
    {
        "key": "normal",
        "title": r"Нормальное $N(0,1)$",
        "sample": lambda n: np.random.normal(0, 1, n),
        "cdf": lambda x: norm.cdf(x, 0, 1),
        "pdf": lambda x: norm.pdf(x, 0, 1),
        "xlim": (-4, 4),
        "bandwidth": silverman_bandwidth,
        "color": "royalblue",
    },

    {
        "key": "cauchy",
        "title": r"Коши $C(0,1)$",
        "sample": lambda n: np.random.standard_cauchy(n),
        "cdf": lambda x: cauchy.cdf(x, 0, 1),
        "pdf": lambda x: cauchy.pdf(x, 0, 1),
        "xlim": (-4, 4),
        "bandwidth": robust_bandwidth,
        "color": "darkorange",
    },

    {
        "key": "laplace",
        "title": r"Лапласа $L(0,1/\sqrt{2})$",
        "sample": lambda n: np.random.laplace(0, 1 / np.sqrt(2), n),
        "cdf": lambda x: laplace.cdf(x, 0, 1 / np.sqrt(2)),
        "pdf": lambda x: laplace.pdf(x, 0, 1 / np.sqrt(2)),
        "xlim": (-4, 4),
        "bandwidth": silverman_bandwidth,
        "color": "seagreen",
    },

    {
        "key": "poisson",
        "title": r"Пуассона $P(10)$",
        "sample": lambda n: np.random.poisson(10, n),
        "cdf": lambda x: poisson.cdf(np.floor(x), 10),
        "pmf": lambda k: poisson.pmf(k, 10),
        "xlim": (6, 14),
        "bandwidth": silverman_bandwidth,
        "discrete": True,
        "color": "firebrick",
    },

    {
        "key": "uniform",
        "title": r"Равномерное $U(-\sqrt{3},\sqrt{3})$",
        "sample": lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n),
        "cdf": lambda x: uniform.cdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
        "pdf": lambda x: uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
        "xlim": (-4, 4),
        "bandwidth": silverman_bandwidth,
        "color": "purple",
    }
]

sample_sizes = [20, 60, 100]

# =========================
# ОСНОВНОЙ ЦИКЛ
# =========================

for dist in distributions:

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    fig.suptitle(
        f'Распределение {dist["title"]}',
        fontsize=16
    )

    xlim = dist["xlim"]

    is_discrete = dist.get("discrete", False)

    for idx, n in enumerate(sample_sizes):

        sample = dist["sample"](n)

        # =====================================
        # ВЕРХНИЙ РЯД — ПЛОТНОСТЬ И KDE
        # =====================================

        ax_density = axes[0, idx]

        if is_discrete:

            bins = np.arange(5.5, 15.5, 1)

            ax_density.hist(
                sample,
                bins=bins,
                density=True,
                alpha=0.35,
                color="gray",
                edgecolor="black",
                label="Гистограмма"
            )

        else:

            if dist["key"] == "cauchy":

                if n == 20:
                    bins = 25
                elif n == 60:
                    bins = 35
                else:
                    bins = 45

            else:
                bins = "fd"

            ax_density.hist(
                sample,
                bins=bins,
                range=xlim,
                density=True,
                alpha=0.35,
                color="gray",
                edgecolor="black",
                label="Гистограмма"
            )

        # KDE

        x_grid = np.linspace(xlim[0], xlim[1], 500)

        h = dist["bandwidth"](sample)

        kde_values = gaussian_kernel_density(
            x_grid,
            sample,
            h
        )

        ax_density.plot(
            x_grid,
            kde_values,
            color="blue",
            linewidth=2,
            label="Ядерная оценка"
        )

        # Теоретическая функция

        if is_discrete:

            k_values = np.arange(6, 15)

            pmf_values = dist["pmf"](k_values)

            ax_density.plot(
                k_values,
                pmf_values,
                "ro-",
                linewidth=1.5,
                markersize=4,
                label="Теоретическая вероятность"
            )

        else:

            pdf_values = dist["pdf"](x_grid)

            ax_density.plot(
                x_grid,
                pdf_values,
                color="red",
                linewidth=2,
                label="Теоретическая плотность"
            )

        ax_density.set_xlim(xlim)

        ax_density.set_xlabel("x")

        ax_density.set_ylabel("Плотность")

        ax_density.set_title(f"n = {n}")

        ax_density.grid(alpha=0.3)

        ax_density.legend(fontsize=8)

        # =====================================
        # НИЖНИЙ РЯД — ЭФР
        # =====================================

        ax_cdf = axes[1, idx]

        sorted_sample = np.sort(sample)

        ecdf_y = np.arange(1, n + 1) / n

        ax_cdf.step(
            sorted_sample,
            ecdf_y,
            where="post",
            color="blue",
            linewidth=2,
            label="Эмпирическая ФР"
        )

        if is_discrete:

            x_values = np.arange(6, 15)

            y_values = dist["cdf"](x_values)

            ax_cdf.step(
                x_values,
                y_values,
                where="post",
                color="red",
                linewidth=2,
                label="Теоретическая ФР"
            )

        else:

            x_values = np.linspace(xlim[0], xlim[1], 500)

            y_values = dist["cdf"](x_values)

            ax_cdf.plot(
                x_values,
                y_values,
                color="red",
                linewidth=2,
                label="Теоретическая ФР"
            )

        ax_cdf.set_xlim(xlim)

        ax_cdf.set_ylim(0, 1.05)

        ax_cdf.set_xlabel("x")

        ax_cdf.set_ylabel("F(x)")

        ax_cdf.grid(alpha=0.3)

        ax_cdf.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        outdir / f"{dist['key']}.png",
        dpi=150
    )

    plt.close()

# =========================
# ВЛИЯНИЕ ШИРИНЫ ОКНА
# =========================

# ---------- Равномерное ----------

sample_uniform = np.random.uniform(
    -np.sqrt(3),
    np.sqrt(3),
    100
)

x = np.linspace(-4, 4, 500)

base_h = silverman_bandwidth(sample_uniform)

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
    color="red",
    linewidth=2,
    label="Теоретическая плотность"
)

for coef, style in zip(
    [0.5, 1.0, 2.0],
    ["--", "-", ":"]
):

    h = coef * base_h

    kde = gaussian_kernel_density(x, sample_uniform, h)

    plt.plot(
        x,
        kde,
        linestyle=style,
        linewidth=1.8,
        label=f"KDE, h={coef:.1f}h"
    )

plt.xlim(-4, 4)

plt.xlabel("x")

plt.ylabel("Плотность")

plt.title("Влияние ширины окна (равномерное распределение)")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    outdir / "uniform_bandwidth.png",
    dpi=150
)

plt.close()

# ---------- Лапласа ----------

sample_laplace = np.random.laplace(
    0,
    1 / np.sqrt(2),
    100
)

x = np.linspace(-4, 4, 500)

base_h = silverman_bandwidth(sample_laplace)

plt.figure(figsize=(8, 5))

plt.plot(
    x,
    laplace.pdf(x, 0, 1 / np.sqrt(2)),
    color="red",
    linewidth=2,
    label="Теоретическая плотность"
)

for coef, style in zip(
    [0.5, 1.0, 2.0],
    ["--", "-", ":"]
):

    h = coef * base_h

    kde = gaussian_kernel_density(x, sample_laplace, h)

    plt.plot(
        x,
        kde,
        linestyle=style,
        linewidth=1.8,
        label=f"KDE, h={coef:.1f}h"
    )

plt.xlim(-4, 4)

plt.xlabel("x")

plt.ylabel("Плотность")

plt.title("Влияние ширины окна (распределение Лапласа)")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    outdir / "laplace_bandwidth.png",
    dpi=150
)

plt.close()
