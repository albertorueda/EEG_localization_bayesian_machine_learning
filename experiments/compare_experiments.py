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

La línea vertical discontinua marca size = n_channels (transición sobre -> infradeterminado).
Los experimentos sin resultados se saltan con un aviso.

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
ERR_ACT, ACT_OFF, ERR_TOT, TIME, MED_ACT = range(5)


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
        return (
            abs(x_hat[0] - X_TRUE_0),
            np.linalg.norm(x_hat[1:]),
            float(d[f"{prefix}_norm_diff"]) if f"{prefix}_norm_diff" in d.files else np.nan,
            float(d[f"{prefix}_time"]) if f"{prefix}_time" in d.files else np.nan,
            median_active_error(d) if prefix == "smc" else np.nan,
        )

    res = aggregate(os.path.join(EXP_ROOT, folder, "results"), metrics)
    if res is None:
        return None
    return (*res, info.get("n_channels"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(EXP_ROOT, "comparison.png"))
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="subconjunto de experimentos a dibujar")
    p.add_argument("--logy", action="store_true", help="escala log en los paneles de error")
    p.add_argument("--no_band", action="store_true", help="no dibujar la banda +- 1 std entre seeds")
    args = p.parse_args()

    keys = args.only or list(EXPERIMENTS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    panels = [
        (ERR_ACT, "Error in active source",       r"$|\hat{x}_0 - x_{true,0}|$"),
        (ACT_OFF, "Activity in inactive sources", r"$\|\hat{x}_{1:}\|_2$"),
        (ERR_TOT, "Total error",                  r"$\|\hat{x} - x_{true}\|_2$"),
        (TIME,    "Computation time",             "time [s]"),
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
    for ax in axes[2:]:
        ax.set_xlabel("size (number of sources)")
    if n_channels is not None:
        axes[0].text(n_channels, axes[0].get_ylim()[1], f" n_channels={n_channels}",
                     color="gray", va="top", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out, dpi=200)

    # Tabla resumen (mediana sobre sizes de la media sobre seeds)
    print(f"\n{'method':32s} {'sizes':>5s} {'jobs':>5s} {'err_act':>10s} {'act_off':>10s} {'err_tot':>10s} {'time[s]':>10s}")
    for lab, ns, nj, a, b, c, t in summary:
        print(f"{lab:32s} {ns:5d} {nj:5d} {a:10.4f} {b:10.4f} {c:10.4f} {t:10.2f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
