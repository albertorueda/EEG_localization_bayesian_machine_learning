"""
aggregate_results.py

Lee results/size_XXXX.npz (un fichero por job de hmc_experiment.py) y dibuja, para HMC (posterior mean):
  - error en la fuente activa:      |x_hat[0] - x_true[0]|
  - actividad en fuentes inactivas: ||x_hat[1:]||_2
frente a size, en una sola figura con dos paneles.

Uso:
    python experiments/hmc_map_ini_experiment/aggregate_results.py
    python experiments/hmc_map_ini_experiment/aggregate_results.py --results_dir experiments/hmc_map_ini_experiment/results --out experiments/hmc_map_ini_experiment/error.png
"""
import argparse
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deben coincidir con hmc_experiment.py
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)  # valor real de la fuente 0 en el instante tid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="experiments/hmc_map_ini_experiment/results")
    p.add_argument("--out", default="experiments/hmc_map_ini_experiment/error.png")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.results_dir}/size_*.npz"))
    if not files:
        raise FileNotFoundError(f"No hay ficheros size_*.npz en '{args.results_dir}'")

    rows = []
    for f in files:
        d = np.load(f)
        x_hat = d["hmc_mean"]
        rows.append((int(d["size"]), abs(x_hat[0] - X_TRUE_0), np.linalg.norm(x_hat[1:])))

    rows = np.array(sorted(rows))
    size, err_act, act_off = rows[:, 0], rows[:, 1], rows[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    axes[0].plot(size, err_act, "s-", ms=3, label="HMC (posterior mean)")
    axes[0].set_title("Error in active source")
    axes[0].set_ylabel(r"$|\hat{x}_0 - x_{true,0}|$")

    axes[1].plot(size, act_off, "s-", ms=3, label="HMC (posterior mean)")
    axes[1].set_title("Activity in inactive sources")
    axes[1].set_ylabel(r"$\|\hat{x}_{1:}\|_2$")

    for ax in axes:
        ax.set_xlabel("size")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"{len(size)} jobs -> {args.out}")


if __name__ == "__main__":
    main()
