"""
compare_experiments.py

Junta los resultados (results/size_XXXX_seed_YY.npz) de todos los experimentos de barrido en `size`
y los dibuja en la misma figura para poder compararlos. Para cada size se promedia sobre las seeds
disponibles: linea = media sobre seeds, banda = +- 1 std (se quita con --no_band).

  (a) error en la fuente activa:       |x_hat[0] - x_true[0]|
      (para SMC, ademas, una segunda linea con la mediana de theta_0 | activa)
  (b) actividad en fuentes inactivas:  ||x_hat[1:]||_2
  (c) error total:                     ||x_hat - x_true||_2   (norm_diff guardado por cada job)
  (d) tiempo de cómputo                (escala log)
  (e) probabilidad posterior p(z0=1) frente a z_true0=1
  (f) probabilidad posterior media/máxima de las fuentes inactivas frente a z_true=0

La línea vertical discontinua marca size = n_channels (transición sobre -> infradeterminado).
Los experimentos sin resultados se saltan con un aviso.
Además se genera una figura *_activation con curvas ROC y scatter que comparan
|x_j_mean| frente a p(z=1) como criterio de activación.

"Mediana | activa" = mediana de theta_0 usando solo las particulas con |theta_0| > ACTIVE_THR
(las que no estan en el spike). Si ninguna esta activa -> NaN (esa seed se ignora en la media;
si es NaN en todas las seeds, hueco en la curva).

Uso:
    python experiments/compare_experiments.py
    python experiments/compare_experiments.py --out experiments/comparison.png --logy
    python experiments/compare_experiments.py --only map hmc hmc_inverse_map_ini
    python experiments/compare_experiments.py --no_band
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
from glob import glob

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.seeds import aggregate, plot_band, load_samples, has_samples

# Deben coincidir con los scripts de los experimentos
A = 0.5
TID = 62
FS = 250
X_TRUE_0 = A * np.sin(2 * np.pi * TID / FS)
SPIKE_VAR = 0.001
ACTIVE_THR = 3 * np.sqrt(SPIKE_VAR)          # ~0.095: por encima, la particula no esta en el spike

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))

# clave -> (carpeta, prefijo de las claves en el .npz, etiqueta, color, marcador, estilo de línea)
EXPERIMENTS = {
    "map":                 ("map_experiment",                 "map", "MAP",                          "black",   "o", "-"),
    "hmc":                 ("hmc_experiment",                 "hmc", "HMC",                          "tab:blue", "s", "-"),
    "hmc_map_ini":         ("hmc_map_ini_experiment",         "hmc", "HMC (MAP init)",               "tab:blue", "^", "--"),
    "hmc_inverse":         ("hmc_inverse_experiment",         "hmc", "HMC inverse mass",             "tab:red",  "s", "-"),
    "hmc_inverse_map_ini": ("hmc_inverse_map_ini_experiment", "hmc", "HMC inverse mass (MAP init)",  "tab:red",  "^", "--"),
    "smc":                 ("smc_experiment",                 "smc", "SMC",                          "tab:green", "D", "-"),
}

# columnas de las metricas por fichero
ERR_ACT, ACT_OFF, ERR_TOT, TIME, MED_ACT, PZ1_ACT, PZ1_OFF_MEAN, PZ1_OFF_MAX = range(8)


def median_active_error(d):
    """|median(theta_0 | |theta_0| > ACTIVE_THR) - x_true_0| a partir de las particulas guardadas (SMC)."""
    if not has_samples(d):
        return np.nan
    t0 = load_samples(d)[0][:, 0]
    active = np.abs(t0) > ACTIVE_THR
    return abs(np.median(t0[active]) - X_TRUE_0) if active.any() else np.nan


def load_experiment(folder, prefix):
    """(size, mean, std, n_seeds, n_channels) agregando sobre seeds, o None si no hay resultados."""
    info = {}

    def metrics(d):
        if f"{prefix}_mean" not in d.files:
            return None
        info["n_channels"] = int(d["n_channels"])
        x_hat = d[f"{prefix}_mean"]
        p_z1 = np.asarray(d["p_z1"], dtype=float) if "p_z1" in d.files else None
        if p_z1 is not None:
            pz_active = float(p_z1[0])
            pz_off_mean = float(np.mean(p_z1[1:])) if p_z1.size > 1 else 0.0
            pz_off_max = float(np.max(p_z1[1:])) if p_z1.size > 1 else 0.0
        else:
            pz_active = pz_off_mean = pz_off_max = np.nan
        return (
            abs(x_hat[0] - X_TRUE_0),
            np.linalg.norm(x_hat[1:]),
            float(d[f"{prefix}_norm_diff"]) if f"{prefix}_norm_diff" in d.files else np.nan,
            float(d[f"{prefix}_time"]) if f"{prefix}_time" in d.files else np.nan,
            median_active_error(d) if prefix == "smc" else np.nan,
            pz_active,
            pz_off_mean,
            pz_off_max,
        )

    res = aggregate(os.path.join(EXP_ROOT, folder, "results"), metrics)
    if res is None:
        return None
    return (*res, info.get("n_channels"))


def roc_points(y_true, scores):
    """ROC points and AUC for a binary activation vector."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    tpr, fpr = [], []
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    for threshold in thresholds:
        pred = scores >= threshold
        tp = np.sum(pred & y_true)
        fp = np.sum(pred & ~y_true)
        tpr.append(tp / n_pos if n_pos else np.nan)
        fpr.append(fp / n_neg if n_neg else np.nan)
    tpr = np.asarray(tpr)
    fpr = np.asarray(fpr)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def fixed_metrics(y_true, pred):
    y_true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = np.sum(pred & y_true)
    fp = np.sum(pred & ~y_true)
    tn = np.sum(~pred & ~y_true)
    fn = np.sum(~pred & y_true)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    accuracy = (tp + tn) / len(y_true)
    return accuracy, balanced_accuracy, f1


def best_f1(y_true, scores):
    """Threshold maximizing F1 over the observed score values."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    best = (-1.0, float(scores.min()), None)
    for threshold in np.unique(scores):
        metrics = fixed_metrics(y_true, scores >= threshold)
        if metrics[2] > best[0]:
            best = (metrics[2], float(threshold), metrics)
    return best


def collect_activation_scores(folder, prefix):
    """Collect ground-truth, x_j_mean magnitude and p_z1 for all available jobs."""
    y_all, mean_all, pz_all = [], [], []
    for path in glob(os.path.join(EXP_ROOT, folder, "results", "size_*_seed_*.npz")):
        with np.load(path) as d:
            if "p_z1" not in d.files:
                continue
            pz = np.asarray(d["p_z1"], dtype=float)
            if "x_j_mean" in d.files:
                mean = np.abs(np.asarray(d["x_j_mean"], dtype=float))
            elif f"{prefix}_mean" in d.files:
                mean = np.abs(np.asarray(d[f"{prefix}_mean"], dtype=float))
            else:
                continue
            if mean.size != pz.size:
                continue
            y_all.extend([1] + [0] * (pz.size - 1))
            mean_all.extend(mean)
            pz_all.extend(pz)
    if not y_all:
        return None
    return np.asarray(y_all, dtype=bool), np.asarray(mean_all), np.asarray(pz_all)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(EXP_ROOT, "comparison.png"))
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="subconjunto de experimentos a dibujar")
    p.add_argument("--logy", action="store_true", help="escala log en los paneles de error")
    p.add_argument("--no_band", action="store_true", help="no dibujar la banda +- 1 std entre seeds")
    args = p.parse_args()

    keys = args.only or list(EXPERIMENTS)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    panels = [
        (ERR_ACT, "Error in active source",       r"$|\hat{x}_0 - x_{true,0}|$"),
        (ACT_OFF, "Activity in inactive sources", r"$\|\hat{x}_{1:}\|_2$"),
        (ERR_TOT, "Total error",                  r"$\|\hat{x} - x_{true}\|_2$"),
        (TIME,    "Computation time",             "time [s]"),
        (PZ1_ACT, "Posterior probability Z=1",    r"$p(z_0=1)$"),
        (PZ1_OFF_MEAN, "Inactive posterior probability", r"$mean(p(z_{1:}=1))$"),
    ]

    n_channels = None
    summary = []
    for k in keys:
        folder, prefix, label, color, marker, ls = EXPERIMENTS[k]
        res = load_experiment(folder, prefix)
        if res is None:
            print(f"[aviso] sin resultados para '{k}' ({folder}/results) -> se omite")
            continue
        size, mean, std, n, nc = res
        n_channels = n_channels or nc
        if args.no_band:
            std = np.zeros_like(std)
        seeds_txt = f"{n.min()}" if n.min() == n.max() else f"{n.min()}-{n.max()}"
        for ax, (col, _, _) in zip(axes, panels):
            plot_band(ax, size, mean[:, col], std[:, col], color=color, marker=marker, ls=ls, ms=3.5, lw=1.4,
                      mfc="white" if ls == "--" else color,
                      label=f"{label} (sizes={len(size)}, seeds={seeds_txt})")
        # linea extra en el panel (a): mediana | activa (solo SMC)
        if prefix == "smc" and not np.all(np.isnan(mean[:, MED_ACT])):
            plot_band(axes[0], size, mean[:, MED_ACT], std[:, MED_ACT], color="darkgreen", marker="o", ls="-.",
                      ms=3.5, lw=1.4, mfc="white", label=f"{label} (median | active)")
        if not np.all(np.isnan(mean[:, PZ1_OFF_MAX])):
            plot_band(axes[5], size, mean[:, PZ1_OFF_MAX], std[:, PZ1_OFF_MAX],
                      color=color, marker=marker, ls=":", ms=3.5, lw=1.1, mfc="white",
                      label=f"{label} (max inactive)")
        summary.append((label, len(size), int(n.sum()),
                        *(np.nanmedian(mean[:, c]) for c in (ERR_ACT, ACT_OFF, ERR_TOT, TIME))))

    if not summary:
        raise FileNotFoundError("No se ha encontrado ningún resultado.")

    for ax, (_, title, ylabel) in zip(axes, panels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if n_channels is not None:
            ax.axvline(n_channels, color="gray", ls=":", lw=1.2)
        if args.logy or ax is axes[3]:
            ax.set_yscale("log")
    for ax in axes[4:]:
        ax.set_xlabel("size (number of sources)")
    axes[4].axhline(1.0, color="tab:red", ls="--", lw=1.2, label="real z_0=1")
    axes[5].axhline(0.0, color="tab:red", ls="--", lw=1.2, label="real z_{1:}=0")
    if n_channels is not None:
        axes[0].text(n_channels, axes[0].get_ylim()[1], f" n_channels={n_channels}",
                     color="gray", va="top", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out, dpi=200)

    # Comparacion de criterios para decidir si una fuente esta activada:
    # |x_j_mean| frente a p(z=1).
    criterion_fig, criterion_axes = plt.subplots(1, 2, figsize=(14, 5.5))
    criterion_summary = []
    scatter_plotted = False
    for k in keys:
        folder, prefix, label, color, marker, ls = EXPERIMENTS[k]
        collected = collect_activation_scores(folder, prefix)
        if collected is None:
            continue
        y_true, mean_score, pz_score = collected

        for ax, scores, criterion_label in (
            (criterion_axes[0], mean_score, r"$|x_j|$ mean"),
            (criterion_axes[0], pz_score, r"$p(z=1)$"),
        ):
            fpr, tpr, auc = roc_points(y_true, scores)
            ax.plot(fpr, tpr, marker=marker if criterion_label == r"$p(z=1)$" else None,
                    ms=3.5, lw=1.5, color=color,
                    ls=ls if criterion_label == r"$|x_j|$ mean" else "--",
                    label=f"{label} - {criterion_label} (AUC={auc:.3f})")

        mean_fixed = fixed_metrics(y_true, mean_score >= ACTIVE_THR)
        pz_fixed = fixed_metrics(y_true, pz_score >= 0.5)
        mean_best = best_f1(y_true, mean_score)
        pz_best = best_f1(y_true, pz_score)
        criterion_summary.append((label, mean_fixed, pz_fixed, mean_best, pz_best))

        # Scatter para visualizar directamente la relacion de ambos criterios.
        criterion_axes[1].scatter(mean_score, pz_score, c=np.where(y_true, "tab:red", "tab:gray"),
                                  s=8, alpha=0.25, rasterized=True)
        scatter_plotted = True

    criterion_axes[0].plot([0, 1], [0, 1], "k:", lw=1.0)
    criterion_axes[0].set_title("ROC: activation criterion")
    criterion_axes[0].set_xlabel("False positive rate")
    criterion_axes[0].set_ylabel("True positive rate")
    criterion_axes[0].set_xlim(-0.02, 1.02)
    criterion_axes[0].set_ylim(-0.02, 1.02)
    criterion_axes[0].grid(alpha=0.3)
    criterion_axes[0].legend(loc="lower right", fontsize=8)
    criterion_axes[1].axvline(ACTIVE_THR, color="tab:blue", ls="--", lw=1.2,
                              label=r"$|x_j|$ threshold")
    criterion_axes[1].axhline(0.5, color="tab:green", ls="--", lw=1.2,
                              label=r"$p(z=1)=0.5$")
    criterion_axes[1].set_title(r"$|x_j|$ mean vs $p(z=1)$")
    criterion_axes[1].set_xlabel(r"$|x_j|$ posterior mean")
    criterion_axes[1].set_ylabel(r"$p(z=1)$")
    criterion_axes[1].set_xlim(left=0)
    criterion_axes[1].set_ylim(-0.02, 1.02)
    criterion_axes[1].grid(alpha=0.3)
    if scatter_plotted:
        criterion_axes[1].legend(fontsize=8)
    criterion_fig.tight_layout()
    out_root, out_ext = os.path.splitext(args.out)
    criterion_out = f"{out_root}_activation{out_ext or '.png'}"
    criterion_fig.savefig(criterion_out, dpi=200)

    # Tabla resumen (mediana sobre sizes de la media sobre seeds)
    print(f"\n{'method':32s} {'sizes':>5s} {'jobs':>5s} {'err_act':>10s} {'act_off':>10s} {'err_tot':>10s} {'time[s]':>10s}")
    for lab, ns, nj, a, b, c, t in summary:
        print(f"{lab:32s} {ns:5d} {nj:5d} {a:10.4f} {b:10.4f} {c:10.4f} {t:10.2f}")
    if criterion_summary:
        print("\nActivation criterion comparison "
              "(fixed thresholds: |x_j| >= 3*sqrt(spike_var), p(z=1) >= 0.5)")
        print(f"{'method':32s} {'criterion':20s} {'accuracy':>10s} {'balanced':>10s} {'F1':>10s} {'best F1':>10s} {'best thr':>10s}")
        for lab, mean_fixed, pz_fixed, mean_best, pz_best in criterion_summary:
            print(f"{lab:32s} {'|x_j_mean|':20s} {mean_fixed[0]:10.4f} {mean_fixed[1]:10.4f} "
                  f"{mean_fixed[2]:10.4f} {mean_best[0]:10.4f} {mean_best[1]:10.4f}")
            print(f"{lab:32s} {'p(z=1)':20s} {pz_fixed[0]:10.4f} {pz_fixed[1]:10.4f} "
                  f"{pz_fixed[2]:10.4f} {pz_best[0]:10.4f} {pz_best[1]:10.4f}")
    print(f"\n-> {args.out}")
    print(f"-> {criterion_out}")


if __name__ == "__main__":
    main()
