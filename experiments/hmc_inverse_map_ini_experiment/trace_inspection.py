"""
trace_inspection.py

Inspect the HMC chains (thetas) stored in results/size_XXXX.npz, to see what the
chain looks like before and after HMC starts performing badly.

For each selected size, three panels:
  1. trace of theta_0 (the truly active source), with true value and burn-in line
  2. traces of a few inactive sources
  3. trace of ||theta_{1:}||_2 (activity in inactive sources) + running mean

Usage:
    python trace_inspection.py                       # auto-picks sizes
    python trace_inspection.py --sizes 12 62 82 202 1002
    python trace_inspection.py --n_inactive 5 --out hmc_traces.png
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must match map_vs_hmc
A = 0.5
TID = 62
FS = 250
BURN_IN = 100
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)

DEFAULT_TARGETS = [80]  # last one -> largest available


def available_sizes(results_dir):
    sizes = {}
    for f in glob.glob(os.path.join(results_dir, "size_*.npz")):
        m = re.search(r"size_(\d+)\.npz$", f)
        if m:
            sizes[int(m.group(1))] = f
    if not sizes:
        raise FileNotFoundError(f"No size_*.npz files found in '{results_dir}'")
    return sizes


def closest(sizes, target):
    return min(sizes, key=lambda s: abs(s - target))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="experiments/map_vs_hmc_inverse/results")
    p.add_argument("--sizes", type=int, nargs="*", default=None)
    p.add_argument("--n_inactive", type=int, default=5,
                   help="number of inactive sources to plot in panel 2")
    p.add_argument("--out", default="experiments/map_vs_hmc_inverse/hmc_traces.png")
    args = p.parse_args()

    files = available_sizes(args.results_dir)
    if args.sizes:
        chosen = sorted({closest(files, s) for s in args.sizes})
    else:
        chosen = sorted({closest(files, s) for s in DEFAULT_TARGETS})
    print("Sizes:", chosen)

    fig, axes = plt.subplots(len(chosen), 3, figsize=(16, 3.2 * len(chosen)), squeeze=False)

    for row, size in enumerate(chosen):
        d = np.load(files[size])
        thetas = d["thetas"]                     # (num_iter + 1, size)
        n_it = thetas.shape[0]
        it = np.arange(n_it)

        moved = np.any(np.diff(thetas, axis=0) != 0, axis=1)
        print(f"size={size:5d}  accept_rate={moved.mean():.3f}  "
              f"theta_0 mean={thetas[BURN_IN:, 0].mean():+.4f} (true {X_TRUE_0:+.4f})  "
              f"std={thetas[BURN_IN:, 0].std():.4f}")

        # 1) active source
        ax = axes[row, 0]
        ax.plot(it, thetas[:, 0], lw=0.8, label=r"$\theta_0$")
        ax.axhline(X_TRUE_0, color="k", ls="--", lw=1, label="true value")
        ax.axvline(BURN_IN, color="r", ls=":", lw=1, label="burn-in")
        ax.set_title(f"size={size}: active source")
        ax.set_ylabel(r"$\theta_0$")

        # 2) a few inactive sources
        ax = axes[row, 1]
        n_in = min(args.n_inactive, size - 1)
        for j in range(1, n_in + 1):
            ax.plot(it, thetas[:, j], lw=0.7, alpha=0.8, label=rf"$\theta_{{{j}}}$")
        ax.axvline(BURN_IN, color="r", ls=":", lw=1)
        ax.set_title(f"size={size}: inactive sources")
        ax.set_ylabel(r"$\theta_j$")

        # 3) norm of inactive sources + running mean of that norm
        ax = axes[row, 2]
        off = np.linalg.norm(thetas[:, 1:], axis=1) if size > 1 else np.zeros(n_it)
        ax.plot(it, off, lw=0.8, label=r"$\|\theta_{1:}\|_2$")
        ax.plot(it, np.cumsum(off) / (it + 1), "k--", lw=1, label="running mean")
        ax.axvline(BURN_IN, color="r", ls=":", lw=1)
        ax.set_title(f"size={size}: activity in inactive sources")
        ax.set_ylabel(r"$\|\theta_{1:}\|_2$")

        for ax in axes[row]:
            ax.set_xlabel("HMC iteration")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()