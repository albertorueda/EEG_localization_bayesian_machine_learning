"""
aggregate_results.py

Lee results/size_XXXX_seed_YY.npz (un fichero por job (size, seed) de l1_experiment.py) y dibuja, para L1 (Lasso):
  - error en la fuente activa:      |x_hat[0] - x_true[0]|
  - actividad en fuentes inactivas: ||x_hat[1:]||_2
  - alpha elegido por CV
  - nº de fuentes activas (coeficientes != 0)
frente a size. Linea = media sobre seeds.

Uso:
    python experiments/l1_experiment/aggregate_results.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.seeds import aggregate, plot_band

# Deben coincidir con l1_experiment.py
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)  # valor real de la fuente 0 en el instante tid

HERE = Path(__file__).resolve().parent


def metrics(d):
    x_hat = d["l1_mean"]
    return abs(x_hat[0] - X_TRUE_0), np.linalg.norm(x_hat[1:]), float(d["l1_alpha"]), int(d["l1_active"].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=str(HERE / "results"))
    p.add_argument("--out", default=str(HERE / "error.png"))
    args = p.parse_args()

    res = aggregate(args.results_dir, metrics)
    if res is None:
        raise FileNotFoundError(f"No hay ficheros size_*_seed_*.npz en '{args.results_dir}'")
    size, mean, std, n = res
    label = f"L1 / Lasso [{n.max()} seeds]"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    panels = [
        ("Error in active source",       r"$|\hat{x}_0 - x_{true,0}|$"),
        ("Activity in inactive sources", r"$\|\hat{x}_{1:}\|_2$"),
        ("CV-selected alpha",            r"$\alpha$"),
        ("Number of active sources",     r"$\#\{j: \hat{x}_j \neq 0\}$"),
    ]
    for i, (ax, (title, ylabel)) in enumerate(zip(axes, panels)):
        plot_band(ax, size, mean[:, i], std[:, i], marker="o", ls="-", ms=3, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[2].set_yscale("log")
    for ax in axes[2:]:
        ax.set_xlabel("size")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"{len(size)} sizes, {n.sum()} jobs -> {args.out}")


if __name__ == "__main__":
    main()
