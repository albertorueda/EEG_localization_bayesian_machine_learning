"""
seeds.py

Utilidades compartidas para repetir los experimentos de barrido en `size` con varias seeds.

Cada job del array de LSF corresponde a un par (size, seed):
    job_idx (1-based)  ->  size_idx, seed = divmod(job_idx - 1, N_SEEDS)
de modo que hay len(sizes) * N_SEEDS jobs en total (#BSUB -J nombre[1-len(sizes)*N_SEEDS]).

La seed controla:
  - el ruido de los datos sinteticos:  np.random.default_rng(NOISE_SEED_BASE + seed)
  - la semilla del sampler (HMC / SMC) y de la inicializacion aleatoria
Con seed = 0 se reproducen exactamente los resultados anteriores (ruido 12345, sampler 0).

Ficheros de resultados:  results/size_XXXX_seed_YY.npz
"""
import glob
import os
import re
import sys
import warnings

import numpy as np

N_SEEDS = 10
NOISE_SEED_BASE = 12345

# Para no guardar la cadena/particulas completas (num_iter x size, float64) en cada job:
N_KEEP = 6                 # columnas guardadas: theta_0 + 5 fuentes inactivas (lo que dibuja trace_inspection)
FULL_CHAIN_SEEDS = (0,)    # seeds para las que, ademas, se guarda la cadena completa (float32, solo para inspeccionar trazas)

_FNAME_RE = re.compile(r"size_(\d+)_seed_(\d+)\.npz$")


def compact_samples(thetas, seed):
    """
    Resumen compacto de una cadena HMC (num_iter+1, size) o de las particulas SMC (N, size):
      thetas_head : (n, min(N_KEEP, size)) float64   primeras fuentes
      off_norm    : (n,) float64                     ||theta_{1:}||_2 por iteracion / particula
      moved       : (n-1,) bool                      la cadena se movio entre iteraciones (aceptacion HMC)
      thetas      : (n, size) float32                solo si seed in FULL_CHAIN_SEEDS
    Uso:  np.savez_compressed(path, ..., **compact_samples(thetas, seed))
    """
    th = np.asarray(thetas, dtype=np.float64)
    out = dict(
        thetas_head=th[:, :N_KEEP].copy(),
        off_norm=(np.linalg.norm(th[:, 1:], axis=1) if th.shape[1] > 1 else np.zeros(len(th))),
        moved=np.any(np.diff(th, axis=0) != 0, axis=1),
    )
    if seed in FULL_CHAIN_SEEDS:
        out["thetas"] = th.astype(np.float32)
    return out


def load_samples(d):
    """(thetas_head, off_norm, moved) de un .npz, en formato compacto o antiguo (thetas completo)."""
    if "thetas_head" in d.files:
        return d["thetas_head"], d["off_norm"], d["moved"]
    th = d["thetas"]
    off = np.linalg.norm(th[:, 1:], axis=1) if th.shape[1] > 1 else np.zeros(len(th))
    return th[:, :N_KEEP], off, np.any(np.diff(th, axis=0) != 0, axis=1)


def has_samples(d):
    return "thetas_head" in d.files or "thetas" in d.files


def parse_job(argv, sizes, job_name, n_seeds=N_SEEDS):
    """Devuelve (job_idx, size, seed). Sin argumentos imprime cuantos jobs hacen falta y sale."""
    n_jobs = len(sizes) * n_seeds
    if len(argv) < 2:
        print(f"n_sizes={len(sizes)}, n_seeds={n_seeds}, n_jobs={n_jobs}  ->  #BSUB -J {job_name}[1-{n_jobs}]")
        sys.exit(0)
    job_idx = int(argv[1])
    if not 1 <= job_idx <= n_jobs:
        raise ValueError(f"job_idx={job_idx} fuera de rango [1, {n_jobs}]")
    size_idx, seed = divmod(job_idx - 1, n_seeds)
    return job_idx, sizes[size_idx], seed


def noise_rng(seed):
    return np.random.default_rng(NOISE_SEED_BASE + seed)


def result_path(results_dir, size, seed):
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"size_{size:04d}_seed_{seed:02d}.npz")


def list_results(results_dir):
    """{size: {seed: path}} con todos los ficheros size_XXXX_seed_YY.npz del directorio."""
    out = {}
    for f in glob.glob(os.path.join(results_dir, "size_*_seed_*.npz")):
        m = _FNAME_RE.search(f)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
    return out


def aggregate(results_dir, metric_fn):
    """
    Aplica metric_fn(npz) -> tupla de metricas (o None para saltar) a cada fichero y agrupa por size.

    Devuelve (size, mean, std, n) con
        size: (S,)        mean, std: (S, M)  (nanmean / nanstd sobre seeds)        n: (S,) nº de seeds
    o None si no hay resultados.
    """
    files = list_results(results_dir)
    sizes, means, stds, ns = [], [], [], []
    for size in sorted(files):
        vals = []
        for seed in sorted(files[size]):
            with np.load(files[size][seed]) as d:
                v = metric_fn(d)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        vals = np.asarray(vals, dtype=float)
        sizes.append(size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # columnas todo-NaN -> NaN sin aviso
            means.append(np.nanmean(vals, axis=0))
            stds.append(np.nanstd(vals, axis=0))
        ns.append(len(vals))
    if not sizes:
        return None
    return np.array(sizes), np.array(means), np.array(stds), np.array(ns)


def plot_band(ax, x, mean, std, **kw):
    """Linea con la media sobre seeds y banda +- 1 std."""
    line, = ax.plot(x, mean, **kw)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15, lw=0)
    return line
