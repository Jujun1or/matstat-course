import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats


STAT_KEYS = ["mean", "median", "zR", "zQ", "zTr"]
STAT_NAMES_RUS = [
    "Среднее",
    "Медиана",
    r"$z_R$",
    r"$z_Q$",
    "Усечённое среднее",
]


def trimmed_mean_10(sample: np.ndarray) -> float:
    return float(stats.trim_mean(sample, 0.1))


def compute_statistics(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample)

    mean = float(np.mean(sample))
    median = float(np.median(sample))
    z_r = 0.5 * (float(np.min(sample)) + float(np.max(sample)))

    q1, q3 = np.percentile(sample, [25, 75])
    z_q = 0.5 * (float(q1) + float(q3))

    z_tr = trimmed_mean_10(sample)

    return np.array([mean, median, z_r, z_q, z_tr], dtype=float)


def estimate_E_D(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e_z = np.mean(values, axis=0)
    e_z2 = np.mean(values ** 2, axis=0)
    d_z = e_z2 - e_z ** 2
    return e_z, d_z


# ===== ФОРМАТИРОВАНИЕ =====

def fmt4(x: float) -> str:
    value = f"{x:.4f}"
    return value.replace("-0.0000", "0.0000")


def fmt3(x: float) -> str:
    value = f"{x:.3f}"
    return value.replace("-0.000", "0.000")


# ===== CSV =====

def write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Распределение", "n", "Статистика", "E(z)", "D(z)", "sqrt(D(z))"])
        writer.writerows(rows)


# ===== LATEX =====

def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    ns: list[int],
    E: dict[int, np.ndarray],
    D: dict[int, np.ndarray],
    sqrtD: dict[int, np.ndarray],
    rounded: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = fmt3 if rounded else fmt4

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{|l|cc|cc|cc|}")
    lines.append(r"\hline")
    lines.append(r"\multirow{2}{*}{Статистика} & \multicolumn{2}{c|}{$n=10$} & \multicolumn{2}{c|}{$n=100$} & \multicolumn{2}{c|}{$n=1000$} \\")
    lines.append(r"\cline{2-7}")
    lines.append(r" & $E(z)$ & $\sqrt{D(z)}$ & $E(z)$ & $\sqrt{D(z)}$ & $E(z)$ & $\sqrt{D(z)}$ \\")
    lines.append(r"\hline")

    for i, stat_name in enumerate(STAT_NAMES_RUS):
        row = [stat_name]
        for n in ns:
            row.append(formatter(E[n][i]))
            row.append(formatter(sqrtD[n][i]))
        lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ===== MAIN =====

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default="report")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    repeats = args.repeats
    ns = [10, 100, 1000]

    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    results_dir = outdir / "results"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    distributions = [
        (
            "normal",
            "Нормальное",
            r"нормального распределения $N(0,1)$",
            lambda n: rng.normal(0.0, 1.0, size=n),
        ),
        (
            "cauchy",
            "Коши",
            r"распределения Коши $C(0,1)$",
            lambda n: rng.standard_cauchy(size=n),
        ),
        (
            "laplace",
            "Лапласа",
            r"распределения Лапласа $L(0, 1/\sqrt{2})$",
            lambda n: rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=n),
        ),
        (
            "poisson",
            "Пуассона",
            r"распределения Пуассона $P(10)$",
            lambda n: rng.poisson(lam=10.0, size=n),
        ),
        (
            "uniform",
            "Равномерное",
            r"равномерного распределения $U(-\sqrt{3}, \sqrt{3})$",
            lambda n: rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n),
        ),
    ]

    csv_rows: list[list[str]] = []

    for key, rus_name, title_latex, generator in distributions:
        E: dict[int, np.ndarray] = {}
        D: dict[int, np.ndarray] = {}
        sqrtD: dict[int, np.ndarray] = {}

        for n in ns:
            values = np.zeros((repeats, len(STAT_KEYS)), dtype=float)

            for r in range(repeats):
                sample = generator(n)
                values[r] = compute_statistics(sample)

            e_z, d_z = estimate_E_D(values)
            sd_z = np.sqrt(d_z)

            E[n] = e_z
            D[n] = d_z
            sqrtD[n] = sd_z

            for i, stat_name in enumerate(STAT_NAMES_RUS):
                csv_rows.append([
                    rus_name,
                    str(n),
                    stat_name,
                    fmt4(e_z[i]),
                    fmt4(d_z[i]),
                    fmt4(sd_z[i]),
                ])

        write_latex_table(
            tables_dir / f"{key}_table_precise.tex",
            caption=rf"Характеристики положения для {title_latex}.",
            label=f"tab:{key}_precise",
            ns=ns,
            E=E,
            D=D,
            sqrtD=sqrtD,
            rounded=False,
        )

        write_latex_table(
            tables_dir / f"{key}_table_rounded.tex",
            caption=rf"Характеристики положения для {title_latex} (округление до трёх знаков).",
            label=f"tab:{key}_rounded",
            ns=ns,
            E=E,
            D=D,
            sqrtD=sqrtD,
            rounded=True,
        )

    write_csv_rows(results_dir / "lab2_results.csv", csv_rows)

    print("Готово.")
    print(f"CSV: {results_dir / 'lab2_results.csv'}")
    print(f"LaTeX-таблицы: {tables_dir}")


if __name__ == "__main__":
    main()