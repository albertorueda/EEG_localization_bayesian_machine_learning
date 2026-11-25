"""
activation.py

Evaluacion de p(z=1) (probabilidad posterior de activacion, guardada como `p_z1` en cada .npz)
como detector de fuentes activas. Ground-truth de todos los experimentos de barrido en `size`:
solo la fuente 0 esta activa (z_0 = 1, z_{1:} = 0).

  - roc_points / fixed_metrics / best_f1 : metricas de clasificacion binaria (O(N log N))
  - pz1_metrics(d)                       : metricas por job (fichero .npz)
  - plot_pz1(results_dir, out, label)    : figura de rendimiento de p(z=1) frente a size + resumen global
"""
import glob
import os

import numpy as np

from lib.seeds import aggregate, plot_band

PZ_THR = 0.5   # umbral fijo: fuente activa si p(z=1) >= 0.5


# ---------------------------------------------------------------- metricas binarias
def _cum_counts(y_true, scores):
    """Thresholds (unique scores, descending) and TP/FP counts for pred = scores >= threshold. O(N log N)."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # last index of each group of equal scores
    return s[last], tp[last], fp[last], int(y.sum()), int(len(y) - y.sum())


def roc_points(y_true, scores):
    """ROC points and AUC for a binary activation vector."""
    _, tp, fp, n_pos, n_neg = _cum_counts(y_true, scores)
    tpr = np.r_[0.0, tp / n_pos] if n_pos else np.full(len(tp) + 1, np.nan)
    fpr = np.r_[0.0, fp / n_neg] if n_neg else np.full(len(fp) + 1, np.nan)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def metrics_from_counts(tp, fp, tn, fn):
    """(accuracy, balanced accuracy, F1)."""
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return accuracy, balanced_accuracy, f1


def fixed_metrics(y_true, pred):
    y_true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = np.sum(pred & y_true)
    fp = np.sum(pred & ~y_true)
    tn = np.sum(~pred & ~y_true)
    fn = np.sum(~pred & y_true)
    return metrics_from_counts(tp, fp, tn, fn)


def best_f1(y_true, scores):
    """Threshold maximizing F1 over the observed score values (smallest threshold on ties)."""
    thr, tp, fp, n_pos, n_neg = _cum_counts(y_true, scores)
    fn = n_pos - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    i = len(f1) - 1 - int(np.argmax(f1[::-1]))            # thr is descending -> last max = smallest thr
    metrics = metrics_from_counts(tp[i], fp[i], n_neg - fp[i], fn[i])
    return metrics[2], float(thr[i]), metrics


# ---------------------------------------------------------------- p(z=1) por job
PZ_COLS = ("pz_active", "pz_off_mean", "pz_off_max", "tpr", "fpr", "f1", "auc", "n_detected")
(PZ_ACTIVE, PZ_OFF_MEAN, PZ_OFF_MAX, PZ_TPR, PZ_FPR, PZ_F1, PZ_AUC, PZ_NDET) = range(len(PZ_COLS))


def pz1_metrics(d, thr=PZ_THR):
    """
    Metricas de p(z=1) de un job (o None si el .npz no tiene p_z1):
      pz_active   p(z_0=1)
      pz_off_*    media / max de p(z_{1:}=1)
      tpr, fpr    con el umbral fijo p(z=1) >= thr  (tpr es 0 o 1: solo hay una fuente activa)
      f1          con el mismo umbral
      auc         P(p(z_0=1) > p(z_j=1)) para j inactiva (empates cuentan 1/2); NaN si size=1
      n_detected  nº de fuentes con p(z=1) >= thr
    """
    if "p_z1" not in d.files:
        return None
    p = np.asarray(d["p_z1"], dtype=float)
    p0, off = p[0], p[1:]
    pred = p >= thr
    tp, fp = int(pred[0]), int(pred[1:].sum())
    fn, tn = 1 - tp, off.size - fp
    f1 = metrics_from_counts(tp, fp, tn, fn)[2]
    auc = float(np.mean((p0 > off) + 0.5 * (p0 == off))) if off.size else np.nan
    return (
        p0,
        float(off.mean()) if off.size else np.nan,
        float(off.max()) if off.size else np.nan,
        float(tp),
        fp / off.size if off.size else np.nan,
        f1,
        auc,
        int(pred.sum()),
    )


def pooled_scores(results_dir):
    """(y_true, p_z1) concatenando todos los jobs, o None."""
    ys, ps = [], []
    for path in glob.glob(os.path.join(results_dir, "size_*_seed_*.npz")):
        with np.load(path) as d:
            if "p_z1" not in d.files:
                continue
            p = np.asarray(d["p_z1"], dtype=float)
        ys.append(np.r_[True, np.zeros(p.size - 1, dtype=bool)])
        ps.append(p)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------- figura por experimento
def plot_pz1(results_dir, out, label, marker="o", thr=PZ_THR):
    """
    Figura 2x2 del rendimiento de p(z=1) frente a size (media sobre seeds) y resumen global por consola.
    Devuelve False (sin figura) si los resultados no tienen p_z1.
    """
    import matplotlib.pyplot as plt

    res = aggregate(results_dir, lambda d: pz1_metrics(d, thr))
    if res is None:
        print(f"[aviso] sin p_z1 en '{results_dir}' -> no se genera {out}")
        return False
    size, mean, std, n = res
    n_channels = None
    files = glob.glob(os.path.join(results_dir, "size_*_seed_*.npz"))
    if files:
        with np.load(files[0]) as d:
            n_channels = int(d["n_channels"]) if "n_channels" in d.files else None

    kw = dict(marker=marker, ms=3, lw=1.3)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax = axes[0, 0]
    plot_band(ax, size, mean[:, PZ_ACTIVE], std[:, PZ_ACTIVE], ls="-", label=r"$p(z_0=1)$ (active)", **kw)
    plot_band(ax, size, mean[:, PZ_OFF_MEAN], std[:, PZ_OFF_MEAN], ls="--", label=r"mean $p(z_{1:}=1)$", **kw)
    plot_band(ax, size, mean[:, PZ_OFF_MAX], std[:, PZ_OFF_MAX], ls=":", label=r"max $p(z_{1:}=1)$", **kw)
    ax.axhline(thr, color="gray", ls="--", lw=1)
    ax.set_title("Posterior activation probability")
    ax.set_ylabel(r"$p(z=1)$")
    ax.set_ylim(-0.03, 1.03)

    ax = axes[0, 1]
    plot_band(ax, size, mean[:, PZ_TPR], std[:, PZ_TPR], ls="-", label="TPR (active detected)", **kw)
    plot_band(ax, size, mean[:, PZ_FPR], std[:, PZ_FPR], ls="--", label="FPR (inactive detected)", **kw)
    plot_band(ax, size, mean[:, PZ_F1], std[:, PZ_F1], ls="-.", label="F1", **kw)
    ax.set_title(f"Detection with $p(z=1) \\geq {thr}$")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.03, 1.03)

    ax = axes[1, 0]
    plot_band(ax, size, mean[:, PZ_AUC], std[:, PZ_AUC], ls="-", label="AUC per job", **kw)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title(r"Ranking: $P(p(z_0=1) > p(z_j=1))$, $j$ inactive")
    ax.set_ylabel("AUC")
    ax.set_ylim(-0.03, 1.03)

    ax = axes[1, 1]
    plot_band(ax, size, mean[:, PZ_NDET], std[:, PZ_NDET], ls="-", label="detected", **kw)
    ax.axhline(1, color="tab:red", ls="--", lw=1, label="true (1)")
    ax.set_title(f"Number of sources with $p(z=1) \\geq {thr}$")
    ax.set_ylabel("# sources")
    ax.set_ylim(bottom=0)

    for a in axes.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        if n_channels is not None:
            a.axvline(n_channels, color="gray", ls=":", lw=1.2)
    for a in axes[1]:
        a.set_xlabel("size (number of sources)")
    fig.suptitle(f"{label}: p(z=1) as activation detector [{n.max()} seeds]")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    pooled = pooled_scores(results_dir)
    if pooled is not None:
        y, p = pooled
        acc, bal, f1 = fixed_metrics(y, p >= thr)
        _, _, auc = roc_points(y, p)
        bf1, bthr, _ = best_f1(y, p)
        print(f"p(z=1) pooled over {n.sum()} jobs: accuracy={acc:.4f}  balanced={bal:.4f}  "
              f"F1@{thr}={f1:.4f}  AUC={auc:.4f}  best F1={bf1:.4f} (thr={bthr:.4f})")
    print(f"-> {out}")
    return True
