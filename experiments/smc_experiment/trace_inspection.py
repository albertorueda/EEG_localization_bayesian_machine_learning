"""
trace_inspection.py

Inspect the SMC chains (thetas) stored in results/size_XXXX.npz, to see what the
chain looks like before and after SMC starts performing badly.

For each selected size, three panels:
  1. trace of theta_0 (the truly active source), with true value and burn-in line
  2. traces of a few inactive sources
  3. trace of ||theta_{1:}||_2 (activity in inactive sources) + running mean

Usage:
    python trace_inspection.py                       # auto-picks sizes
    python trace_inspection.py --sizes 12 62 82 202 1002
    python trace_inspection.py --n_inactive 5 --out smc_traces.png
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from lib.hmc import HMC
import jax.numpy as jnp

from lib.distributions import SpikeAndSlabPrior, Posterior
from lib.map_solution import get_map_estimate
import mne

slab_var, spike_var, slab_prob = 0.1, 0.001, 0.05
SpikeAndSlab = SpikeAndSlabPrior(slab_var=slab_var, spike_var=spike_var, slab_prob=slab_prob)

# Must match map_vs_smc_experiment.py
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)

DEFAULT_TARGETS = [50, 100, 150]  # last one -> largest available

montage = mne.channels.make_standard_montage("colin27_1020")
chs = [c for c in montage.ch_names if c not in ["T3", "T4", "T5", "T6"]]
info = mne.create_info(chs, sfreq=250.0, ch_types="eeg")
info.set_montage(montage)
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
src = mne.setup_volume_source_space(sphere=sphere, pos=15.0, exclude=30.0, add_interpolator=False)
fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False, eeg=True, mindist=5.0)
L = fwd["sol"]["data"]

n_channels, n_sources = L.shape
size = 100
# ---------- Datos sintéticos (misma semilla en TODOS los jobs) ----------
tid = 62
fs = int(info["sfreq"])
t = np.linspace(0, 1, fs, endpoint=False)
A = 0.5
X = np.zeros((n_sources, fs))
X[0] = A * np.sin(2 * np.pi * t)

beta = 4
rng = np.random.default_rng(12345)   # <- fija: el ruido es el mismo en todos los jobs
e = rng.normal(0, 1/beta, size=(n_channels, fs))

# ---------- Problema con `size` fuentes y TODOS los sensores ----------
L_small = jnp.array(L[:, 0:size])    # (n_channels, size)
x_small = X[0:size]                  # (size, fs); la fuente 0 siempre incluida
Y = L_small @ x_small + e            # idéntico en todos los jobs (solo la fuente 0 está activa)
y_tid = Y[:, tid]
post = Posterior(prior_distribution=SpikeAndSlab, L=L_small, Y=Y, beta=beta, y_tid=y_tid)

map_solution = get_map_estimate(post.log_posterior, theta0=jnp.zeros(size))
map_solution = np.asarray(map_solution).ravel()
print(f"MAP: {map_solution}", flush=True)
map_norm_diff = float(np.linalg.norm(map_solution - x_small[:, tid]))

# ---------- HMC ----------
theta_map = jnp.expand_dims(map_solution, axis=0)
thetas_warmup = HMC(post.log_posterior, 250, theta_map, num_leapfrog_steps=250, step_size=1e-5, seed=0)
Sigma_est = jnp.cov(thetas_warmup.T) + 1e-6 * jnp.eye(thetas_warmup.shape[1])

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
    p.add_argument("--results_dir", default="experiments/smc_experiment/results")
    p.add_argument("--sizes", type=int, nargs="*", default=None)
    p.add_argument("--n_inactive", type=int, default=5,
                   help="number of inactive sources to plot in panel 2")
    p.add_argument("--out", default="experiments/smc_experiment/smc_traces.png")
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
        thetas = d["thetas"]
        x = jnp.asarray(thetas[-1][0:1])
        for eps in [5e-2, 1e-2, 1e-3, 1e-4]:
            print("eps =", eps)
            HMC(post.log_posterior, 50, x, num_leapfrog_steps=20, step_size=eps, Sigma=Sigma_est)
        thetas_all = d["thetas"]
        thetas = thetas_all.mean(axis=1)
        n_it = thetas.shape[0]
        it = np.arange(n_it)
        estimation_x0 = float(thetas[-1, 0])

        moved = np.any(np.diff(thetas, axis=0) != 0, axis=1)
        print(f"size={size:5d}  accept_rate={moved.mean():.3f}  "
              f"theta_0={estimation_x0:+.4f} (true {X_TRUE_0:+.4f})")

        # 1) active source
        ax = axes[row, 0]
        ax.plot(it, thetas[:, 0], lw=0.8, label=r"$\theta_0$")
        ax.axhline(X_TRUE_0, color="k", ls="--", lw=1, label="true value")
        ax.set_title(f"size={size}: active source")
        ax.set_ylabel(r"$\theta_0$")

        # 2) a few inactive sources
        ax = axes[row, 1]
        n_in = min(args.n_inactive, size - 1)
        for j in range(1, n_in + 1):
            ax.plot(it, thetas[:, j], lw=0.7, alpha=0.8, label=rf"$\theta_{{{j}}}$")
        ax.set_title(f"size={size}: inactive sources")
        ax.set_ylabel(r"$\theta_j$")

        # 3) norm of inactive sources + running mean of that norm
        ax = axes[row, 2]
        off = np.linalg.norm(thetas[:, 1:], axis=1) if size > 1 else np.zeros(n_it)
        ax.plot(it, off, lw=0.8, label=r"$\|\theta_{1:}\|_2$")
        ax.plot(it, np.cumsum(off) / (it + 1), "k--", lw=1, label="running mean")
        ax.set_title(f"size={size}: activity in inactive sources")
        ax.set_ylabel(r"$\|\theta_{1:}\|_2$")

        for ax in axes[row]:
            ax.set_xlabel("SMC iteration")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()