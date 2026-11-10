"""
aggregate_results.py

Lee results/size_XXXX.npz (un fichero por job de map_vs_smc_experiment.py) y dibuja, para SMC:
  - error en la fuente activa:      |x_hat[0] - x_true[0]|   (media posterior y mediana | activa)
  - actividad en fuentes inactivas: ||x_hat[1:]||_2           (media posterior)
frente a size, en una sola figura con dos paneles.

"Mediana | activa" = mediana de theta_0 usando solo las particulas con |theta_0| > ACTIVE_THR
(las que no estan en el spike). Si ninguna particula esta activa, se deja NaN (hueco en la curva).

Uso:
    python experiments/smc_experiment/aggregate_results.py
    python experiments/smc_experiment/aggregate_results.py --results_dir experiments/smc_experiment/results --out experiments/smc_experiment/error.png
"""
import argparse
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deben coincidir con map_vs_smc_experiment.py
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)  # valor real de la fuente 0 en el instante tid
SPIKE_VAR = 0.001
ACTIVE_THR = 3 * np.sqrt(SPIKE_VAR)          # ~0.095: por encima, la particula no esta en el spike


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="experiments/smc_experiment/results")
    p.add_argument("--out", default="experiments/smc_experiment/error.png")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.results_dir}/size_*.npz"))
    if not files:
        raise FileNotFoundError(f"No hay ficheros size_*.npz en '{args.results_dir}'")

    rows = []
    for f in files:
        d = np.load(f)
        x_hat = d["smc_mean"]
        t0 = d["thetas"][:, 0]                       # particulas de la fuente 0
        active = np.abs(t0) > ACTIVE_THR
        med_act = np.median(t0[active]) if active.any() else np.nan
        rows.append((int(d["size"]),
                     abs(x_hat[0] - X_TRUE_0),
                     abs(med_act - X_TRUE_0),
                     np.linalg.norm(x_hat[1:])))

    rows = np.array(sorted(rows))
    size, err_mean, err_med, act_off = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    axes[0].plot(size, err_mean, "s-", ms=3, label="SMC (posterior mean)")
    axes[0].plot(size, err_med, "o-", ms=3, label="SMC (median | active)")
    axes[0].set_title("Error in active source")
    axes[0].set_ylabel(r"$|\hat{x}_0 - x_{true,0}|$")

    axes[1].plot(size, act_off, "s-", ms=3, label="SMC (posterior mean)")
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