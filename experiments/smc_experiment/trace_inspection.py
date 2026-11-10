"""
trace_inspection.py  (SMC version)

An SMC sampler has no chain, so there are no traces. What is stored in results/size_XXXX.npz is:
    thetas : (N, size)  final equally weighted particles (posterior sample)
    lam, ess, acc, logZ : per-stage history of the tempering

For each selected size one figure with two rows:

  Row 1 (is the sampler healthy?)
    1. tempering schedule: lambda vs stage (log y)
    2. ESS before resampling per stage, with the target ESS_min = ess_frac * N
    3. HMC acceptance per stage (aim for 0.6-0.9; adjust step_size otherwise)

  Row 2 (what does the posterior look like?)
    4. histogram of theta_0 (truly active source), true value and posterior mean
    5. histograms of a few inactive sources
    6. histogram of ||theta_{1:}||_2 per particle (activity in inactive sources)

Usage:
    python trace_inspection.py
    python trace_inspection.py --sizes 50 100 150
    python trace_inspection.py --n_inactive 5 --ess_frac 0.5
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must match the experiment script
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)

DEFAULT_TARGETS = [100]


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


def plot_size(size, path, n_inactive, ess_frac, out):
    d = np.load(path)
    thetas = d["thetas"]                        # (N, size)
    N = thetas.shape[0]
    lam = np.asarray(d["lam"])[1:]              # lambda after each stage
    ess = np.asarray(d["ess"])
    acc = np.asarray(d["acc"])
    logZ = np.asarray(d["logZ"])
    stages = np.arange(1, len(lam) + 1)

    t0 = thetas[:, 0]
    off = np.linalg.norm(thetas[:, 1:], axis=1) if size > 1 else np.zeros(N)
    print(f"size={size:5d}  N={N}  stages={len(lam)}  logZ={logZ[-1]:.2f}  mean acc={acc.mean():.2f} "
          f"(min {acc.min():.2f})  theta_0 mean={t0.mean():+.4f} std={t0.std():.4f} (true {X_TRUE_0:+.4f})  "
          f"||theta_1:|| mean={off.mean():.4f}")
    
    active = np.abs(t0) > 0.1                     # umbral ~3 sigma del spike (sqrt(0.001)=0.032)
    print(f"P(theta_0 active) = {active.mean():.2f}")
    print(f"median | active   = {np.median(t0[active]):+.4f}  (true {X_TRUE_0:+.4f})")

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))

    # 1) tempering schedule
    ax = axes[0, 0]
    ax.plot(stages, lam, "o-", ms=3, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("stage"); ax.set_ylabel(r"$\lambda$")
    ax.set_title(f"size={size}: tempering schedule ({len(lam)} stages)")

    # 2) ESS
    ax = axes[0, 1]
    ax.plot(stages, ess, lw=1, label="ESS")
    ax.axhline(ess_frac * N, color="r", ls="--", lw=1, label=rf"ESS$_{{min}}$ = {ess_frac}·N")
    ax.set_ylim(0, N)
    ax.set_xlabel("stage"); ax.set_ylabel("ESS")
    ax.set_title("Effective sample size per stage")
    ax.legend(fontsize=8)

    # 3) acceptance
    ax = axes[0, 2]
    ax.plot(stages, acc, lw=1)
    ax.axhspan(0.6, 0.9, color="g", alpha=0.12, label="target range")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stage"); ax.set_ylabel("mean acceptance")
    ax.set_title("HMC acceptance per stage")
    ax.legend(fontsize=8)

    # 4) active source
    ax = axes[1, 0]
    ax.hist(t0, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(X_TRUE_0, color="k", ls="--", lw=1.2, label="true value")
    ax.axvline(t0.mean(), color="r", ls="-", lw=1.2, label="posterior mean")
    ax.axvline(np.median(t0[active]), color="g", ls="-.", lw=1.2, label="median | active")
    ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel("particles")
    ax.set_title("Active source")
    ax.legend(fontsize=8)

    # 5) a few inactive sources
    ax = axes[1, 1]
    n_in = min(n_inactive, size - 1)
    for j in range(1, n_in + 1):
        ax.hist(thetas[:, j], bins=40, alpha=0.5, label=rf"$\theta_{{{j}}}$")
    ax.set_xlabel(r"$\theta_j$"); ax.set_ylabel("particles")
    ax.set_title("Inactive sources")
    if n_in > 0:
        ax.legend(fontsize=8)

    # 6) activity in inactive sources
    ax = axes[1, 2]
    ax.hist(off, bins=40, color="tab:orange", alpha=0.8)
    ax.axvline(off.mean(), color="r", lw=1.2, label=f"mean {off.mean():.3f}")
    ax.set_xlabel(r"$\|\theta_{1:}\|_2$"); ax.set_ylabel("particles")
    ax.set_title("Activity in inactive sources")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="experiments/smc_experiment/results")
    p.add_argument("--sizes", type=int, nargs="*", default=None)
    p.add_argument("--n_inactive", type=int, default=5,
                   help="number of inactive sources in panel 5")
    p.add_argument("--ess_frac", type=float, default=0.5, help="must match the SMC call")
    p.add_argument("--out_dir", default="experiments/smc_experiment")
    args = p.parse_args()

    files = available_sizes(args.results_dir)
    targets = args.sizes if args.sizes else DEFAULT_TARGETS
    chosen = sorted({closest(files, s) for s in targets})
    print("Sizes:", chosen)

    os.makedirs(args.out_dir, exist_ok=True)
    for size in chosen:
        out = os.path.join(args.out_dir, f"smc_diagnostics_size_{size:04d}.png")
        plot_size(size, files[size], args.n_inactive, args.ess_frac, out)


if __name__ == "__main__":
    main()