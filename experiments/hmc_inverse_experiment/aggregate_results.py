"""
aggregate_results.py

Lee results/size_XXXX_seed_YY.npz (un fichero por job (size, seed) de hmc_inverse_experiment.py) y dibuja, para HMC inverse mass (posterior mean):
  - error en la fuente activa:      |x_hat[0] - x_true[0]|
  - actividad en fuentes inactivas: ||x_hat[1:]||_2
frente a size. Linea = media sobre seeds, banda = +- 1 std.

Uso:
    python experiments/hmc_inverse_experiment/aggregate_results.py
    python experiments/hmc_inverse_experiment/aggregate_results.py --results_dir experiments/hmc_inverse_experiment/results --out experiments/hmc_inverse_experiment/error.png
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

# Deben coincidir con hmc_inverse_experiment.py
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)  # valor real de la fuente 0 en el instante tid


def metrics(d):
    x_hat = d["hmc_mean"]
    return abs(x_hat[0] - X_TRUE_0), np.linalg.norm(x_hat[1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="experiments/hmc_inverse_experiment/results")
    p.add_argument("--out", default="experiments/hmc_inverse_experiment/error.png")
    args = p.parse_args()

    res = aggregate(args.results_dir, metrics)
    if res is None:
        raise FileNotFoundError(f"No hay ficheros size_*_seed_*.npz en '{args.results_dir}'")
    size, mean, std, n = res
    label = f"HMC inverse mass (posterior mean) [{n.max()} seeds, ±1 std]"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    plot_band(axes[0], size, mean[:, 0], std[:, 0], marker="s", ls="-", ms=3, label=label)
    axes[0].set_title("Error in active source")
    axes[0].set_ylabel(r"$|\hat{x}_0 - x_{true,0}|$")

    plot_band(axes[1], size, mean[:, 1], std[:, 1], marker="s", ls="-", ms=3, label=label)
    axes[1].set_title("Activity in inactive sources")
    axes[1].set_ylabel(r"$\|\hat{x}_{1:}\|_2$")

    for ax in axes:
        ax.set_xlabel("size")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"{len(size)} sizes, {n.sum()} jobs -> {args.out}")


if __name__ == "__main__":
    main()
