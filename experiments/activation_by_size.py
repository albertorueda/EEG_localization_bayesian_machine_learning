"""
activation_by_size.py

Analisis de la deteccion de la fuente activa en funcion de `size`, SIN mezclar jobs.
Cada fichero results/size_XXXX_seed_YY.npz es un problema independiente con z_true = [1, 0, ..., 0],
asi que todas las metricas se calculan POR JOB y luego se agregan por (metodo, size) sobre seeds.

Por que no una ROC global: mezclar todos los sizes/seeds en una sola curva hace que los problemas
faciles (size pequeño) compensen los dificiles (size > n_channels) y esconde justo lo interesante.

Metricas por job:
  - AUC       : fraccion de fuentes inactivas con puntuacion menor que la activa (empates = 0.5).
                Con un unico positivo, esto ES el AUC de la ROC de ese job.
  - top-1     : 1 si la fuente activa tiene la puntuacion MAXIMA (empates cuentan en contra).
  - recall    : 1 si se detecta la fuente activa con umbral fijo (p(z=1)>=0.5, o |x|>=thr si no hay p).
  - FP        : numero de inactivas que superan el umbral fijo.
  - E[FP]     : sum_j p(z_j=1|y) sobre las inactivas (falsos positivos esperados; solo si hay p_z1).
  - F1        : 2TP / (2TP + FP + FN) con el umbral fijo.
  - p(z0=1)   : se dibuja por seed (no solo la media), para ver si es bimodal.

Uso:
    python experiments/activation_by_size.py
    python experiments/activation_by_size.py --only map smc hmc_inverse l1 --table_sizes 50 90 150 300
    python experiments/activation_by_size.py --use_xjmean     # usa x_j_mean en vez de la media posterior
"""
import argparse
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deben coincidir con los experimentos
SPIKE_VAR = 0.001
SLAB_PROB = 0.05
ACTIVE_THR = 3 * np.sqrt(SPIKE_VAR)     # umbral fijo para |x_j|
PZ_THR = 0.5                            # umbral fijo para p(z=1)

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))

# clave -> (carpeta, prefijo en el .npz, etiqueta, color, marcador, estilo)
EXPERIMENTS = {
    "map":                 ("map_experiment",                 "map", "MAP",                         "black",      "o", "-"),
    "hmc":                 ("hmc_experiment",                 "hmc", "HMC",                         "tab:blue",   "s", "-"),
    "hmc_map_ini":         ("hmc_map_ini_experiment",         "hmc", "HMC (MAP init)",              "tab:blue",   "^", "--"),
    "hmc_inverse":         ("hmc_inverse_experiment",         "hmc", "HMC inverse mass",            "tab:red",    "s", "-"),
    "hmc_inverse_map_ini": ("hmc_inverse_map_ini_experiment", "hmc", "HMC inverse mass (MAP init)", "tab:red",    "^", "--"),
    "smc":                 ("smc_experiment",                 "smc", "SMC",                         "tab:green",  "D", "-"),
    "l1":                  ("l1_experiment",                  "l1",  "L1 (Lasso CV)",               "tab:purple", "v", "-"),
}

FILE_RE = re.compile(r"size_(\d+)_seed_(\d+)\.npz$")


# ---------------------------------------------------------------- metricas por job
def auc_one_positive(s):
    """AUC con un unico positivo (indice 0): P(score_0 > score_inactiva) + 0.5 * P(empate)."""
    neg = s[1:]
    if neg.size == 0:
        return np.nan
    return float(((neg < s[0]).sum() + 0.5 * (neg == s[0]).sum()) / neg.size)


def top1(s):
    """1 si la fuente 0 es estrictamente la mayor (empates en contra)."""
    return float(not np.any(s[1:] >= s[0]))


def f1_from_counts(tp, fp, fn):
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den > 0 else 0.0


def job_metrics(x_score, pz, is_l1):
    """x_score: |x_j| (D,), pz: p(z_j=1|y) (D,) o None."""
    out = {"auc_x": auc_one_positive(x_score), "top1_x": top1(x_score)}
    thr_x = 1e-12 if is_l1 else ACTIVE_THR
    det_x = x_score >= thr_x
    out["recall_x"] = float(det_x[0])
    out["fp_x"] = float(det_x[1:].sum())
    out["f1_x"] = f1_from_counts(det_x[0], det_x[1:].sum(), 1 - det_x[0])

    if pz is not None:
        out["auc_p"] = auc_one_positive(pz)
        out["top1_p"] = top1(pz)
        det_p = pz >= PZ_THR
        out["recall_p"] = float(det_p[0])
        out["fp_p"] = float(det_p[1:].sum())
        out["f1_p"] = f1_from_counts(det_p[0], det_p[1:].sum(), 1 - det_p[0])
        out["efp_p"] = float(pz[1:].sum())
        out["pz0"] = float(pz[0])
    return out


# ---------------------------------------------------------------- carga
def collect(folder, prefix, use_xjmean):
    """-> {size: [dict de metricas por seed]}, n_channels"""
    by_size = defaultdict(list)
    n_channels = None
    for path in glob(os.path.join(EXP_ROOT, folder, "results", "size_*_seed_*.npz")):
        m = FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        size = int(m.group(1))
        with np.load(path) as d:
            if use_xjmean and "x_j_mean" in d.files:
                x = d["x_j_mean"]
            elif f"{prefix}_mean" in d.files:
                x = d[f"{prefix}_mean"]
            else:
                continue
            x_score = np.abs(np.asarray(x, dtype=float)).ravel()
            pz = np.asarray(d["p_z1"], dtype=float).ravel() if "p_z1" in d.files else None
            if pz is not None and pz.size != x_score.size:
                pz = None
            if "n_channels" in d.files:
                n_channels = int(d["n_channels"])
        by_size[size].append(job_metrics(x_score, pz, prefix == "l1"))
    return by_size, n_channels


def aggregate(by_size, key):
    """-> sizes, mean, std (sobre seeds) para la metrica `key`; ignora jobs sin esa metrica."""
    sizes, mean, std = [], [], []
    for s in sorted(by_size):
        v = np.array([j[key] for j in by_size[s] if key in j], dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        sizes.append(s)
        mean.append(v.mean())
        std.append(v.std())
    return np.array(sizes), np.array(mean), np.array(std)


def pick(by_size, key):
    """Usa la version con p(z=1) si el metodo la tiene; si no, la version con |x_j|."""
    has_p = any(f"{key}_p" in j for jobs in by_size.values() for j in jobs)
    return (f"{key}_p", "p(z=1)") if has_p else (f"{key}_x", "|x_j|")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP_ROOT, "activation_by_size.png"))
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS))
    ap.add_argument("--no_band", action="store_true")
    ap.add_argument("--use_xjmean", action="store_true",
                    help="usar x_j_mean (media condicional a z=1) como puntuacion |x_j| cuando exista")
    ap.add_argument("--table_sizes", type=int, nargs="*", default=[50, 90, 150, 300])
    args = ap.parse_args()

    keys = args.only or list(EXPERIMENTS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    ax_auc_p, ax_auc_x, ax_top1, ax_rec, ax_fp, ax_pz0 = axes.ravel()

    n_channels = None
    table = []
    for k in keys:
        folder, prefix, label, color, marker, ls = EXPERIMENTS[k]
        by_size, nc = collect(folder, prefix, args.use_xjmean)
        if not by_size:
            print(f"[aviso] sin resultados para '{k}' -> se omite")
            continue
        n_channels = n_channels or nc
        kw = dict(color=color, marker=marker, ls=ls, ms=3.5, lw=1.4,
                  mfc="white" if ls == "--" else color)

        def draw(ax, key, lab_suffix=""):
            s, m, sd = aggregate(by_size, key)
            if s.size == 0:
                return
            ax.plot(s, m, label=f"{label}{lab_suffix}", **kw)
            if not args.no_band:
                ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.12, lw=0)

        draw(ax_auc_p, "auc_p")                                   # solo metodos con p(z=1)
        draw(ax_auc_x, "auc_x")
        top_key, crit = pick(by_size, "top1")
        draw(ax_top1, top_key, f" [{crit}]")
        rec_key, crit = pick(by_size, "recall")
        draw(ax_rec, rec_key, f" [{crit}]")
        fp_key, crit = pick(by_size, "fp")
        draw(ax_fp, fp_key, f" [{crit}]")

        # p(z0=1) por seed + mediana
        xs, ys = [], []
        for s in sorted(by_size):
            for j in by_size[s]:
                if "pz0" in j:
                    xs.append(s)
                    ys.append(j["pz0"])
        if xs:
            ax_pz0.scatter(xs, ys, s=5, alpha=0.25, color=color, rasterized=True)
            xs, ys = np.array(xs), np.array(ys)
            uniq = np.unique(xs)
            ax_pz0.plot(uniq, [np.median(ys[xs == u]) for u in uniq], color=color, lw=1.6,
                        marker=marker, ms=3, mfc="white" if ls == "--" else color, label=f"{label} (mediana)")

        # tabla en sizes cercanos a los pedidos
        avail = np.array(sorted(by_size))
        for ts in args.table_sizes:
            s = int(avail[np.argmin(np.abs(avail - ts))])
            row = {kk: np.nanmean([j[kk] for j in by_size[s] if kk in j] or [np.nan])
                   for kk in ("auc_x", "auc_p", "recall_p", "recall_x", "fp_p", "fp_x", "efp_p", "f1_p", "f1_x", "pz0")}
            table.append((label, s, len(by_size[s]), row))

    if not table:
        raise FileNotFoundError("No se ha encontrado ningun resultado.")

    titles = [
        (ax_auc_p, "AUC por size: ranking con p(z=1)", "AUC (media sobre seeds)"),
        (ax_auc_x, r"AUC por size: ranking con $|x_j|$", "AUC (media sobre seeds)"),
        (ax_top1, "Fuente activa en 1a posicion", "fraccion de seeds"),
        (ax_rec, "Deteccion con umbral fijo (recall)", "fraccion de seeds"),
        (ax_fp, "Falsos positivos con umbral fijo", "# inactivas detectadas"),
        (ax_pz0, r"$p(z_0=1)$ por seed (puntos) y mediana", r"$p(z_0=1)$"),
    ]
    for ax, t, yl in titles:
        ax.set_title(t, fontsize=10)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        if n_channels is not None:
            ax.axvline(n_channels, color="gray", ls=":", lw=1.2)
    for ax in (ax_auc_p, ax_auc_x):
        ax.axhline(0.5, color="k", ls=":", lw=0.8)          # azar
        ax.set_ylim(0.4, 1.02)
    for ax in (ax_top1, ax_rec):
        ax.set_ylim(-0.02, 1.02)
    ax_pz0.axhline(SLAB_PROB, color="tab:orange", ls="--", lw=1.2, label=f"prior pi={SLAB_PROB}")
    ax_pz0.axhline(PZ_THR, color="gray", ls="--", lw=0.8)
    ax_pz0.set_ylim(-0.02, 1.02)
    for ax in (ax_fp, ax_pz0, ax_rec):
        ax.set_xlabel("size (number of sources)")
    for ax in (ax_auc_p, ax_auc_x, ax_top1):
        ax.set_xlabel("size (number of sources)")

    h, l = [], []
    for ax in (ax_top1, ax_rec, ax_fp, ax_pz0):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(args.out, dpi=180)

    # --- tabla
    print(f"\nUmbrales fijos: p(z=1)>={PZ_THR}; |x_j|>={ACTIVE_THR:.3f} (L1: x_j != 0). Medias sobre seeds.")
    print(f"{'method':30s} {'size':>5s} {'seeds':>5s} {'AUC_x':>6s} {'AUC_p':>6s} {'rec_p':>6s} {'rec_x':>6s} "
          f"{'FP_p':>6s} {'FP_x':>6s} {'E[FP]':>7s} {'F1_p':>6s} {'F1_x':>6s} {'p(z0)':>6s}")
    f = lambda v: "   -  " if np.isnan(v) else f"{v:6.2f}"
    for lab, s, ns, r in table:
        print(f"{lab:30s} {s:5d} {ns:5d} {f(r['auc_x'])} {f(r['auc_p'])} {f(r['recall_p'])} {f(r['recall_x'])} "
              f"{f(r['fp_p'])} {f(r['fp_x'])} {f(r['efp_p'])}  {f(r['f1_p'])} {f(r['f1_x'])} {f(r['pz0'])}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()