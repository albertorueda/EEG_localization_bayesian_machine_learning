"""
l1_experiment.py

Igual que map_experiment.py pero estimando la fuente con regularizacion L1 (Lasso de sklearn):

    min_x  1/(2 n_channels) ||y_tid - L x||_2^2 + alpha ||x||_1

alpha se elige por validacion cruzada (LassoCV, 5 folds sobre los canales) en cada job.
Sin intercepto: el modelo directo no lo tiene.

Equivalencia bayesiana: es el MAP con verosimilitud N(Lx, 1/beta^2 I) y prior de Laplace
p(x_j) ∝ exp(-lambda |x_j|), con lambda = alpha * n_channels * beta^2.

Uso:
    python experiments/l1_experiment/l1_experiment.py            # dice cuantos jobs hacen falta
    python experiments/l1_experiment/l1_experiment.py <job_idx>  # 1-based, como $LSB_JOBINDEX
"""
import sys, os
import warnings
import numpy as np
import mne
from time import time
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from lib.seeds import parse_job, noise_rng, result_path

N_FOLDS = 5
N_ALPHAS = 100
MAX_ITER = 50_000

# ---------- Forward model (determinista) ----------
montage = mne.channels.make_standard_montage("colin27_1020")
chs = [c for c in montage.ch_names if c not in ["T3", "T4", "T5", "T6"]]
info = mne.create_info(chs, sfreq=250.0, ch_types="eeg")
info.set_montage(montage)
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
src = mne.setup_volume_source_space(sphere=sphere, pos=15.0, exclude=30.0, add_interpolator=False)
fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False, eeg=True, mindist=5.0)
L = fwd["sol"]["data"]

n_channels, n_sources = L.shape

# `size` = nº de FUENTES (columnas de L). Siempre se usan TODOS los sensores. Mismos sizes que el resto.
sizes = sorted(set(range(5, 301, 5)) | {300, 1})

# Cada job = un par (size, seed): len(sizes) * N_SEEDS jobs en total.
job_idx, size, seed = parse_job(sys.argv, sizes, job_name="l1")   # $LSB_JOBINDEX (1-based)
print(f"Job {job_idx}: n_channels={n_channels}, n_src(size)={size}, seed={seed}", flush=True)

# ---------- Datos sintéticos (el ruido depende de la seed; mismo ruido para todos los sizes) ----------
tid = 62
fs = int(info["sfreq"])
t = np.linspace(0, 1, fs, endpoint=False)
A = 0.5
X = np.zeros((n_sources, fs))
X[0] = A * np.sin(2 * np.pi * t)

beta = 4
rng = noise_rng(seed)               # seed=0 -> default_rng(12345), como en el resto de experimentos
e = rng.normal(0, 1/beta, size=(n_channels, fs))

# ---------- Problema con `size` fuentes y TODOS los sensores ----------
L_small = L[:, 0:size]               # (n_channels, size)
x_small = X[0:size]                  # (size, fs); la fuente 0 siempre incluida
Y = L_small @ x_small + e
y_tid = Y[:, tid]

# ---------- L1 (Lasso con alpha por CV) ----------
t0 = time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    lasso = LassoCV(alphas=N_ALPHAS, cv=N_FOLDS, fit_intercept=False, max_iter=MAX_ITER,
                    random_state=seed).fit(L_small, y_tid)
l1_time = time() - t0
l1_solution = np.asarray(lasso.coef_, dtype=float)
l1_alpha = float(lasso.alpha_)
l1_norm_diff = float(np.linalg.norm(l1_solution - x_small[:, tid]))
l1_active = l1_solution != 0               # soporte estimado (Lasso da ceros exactos)

# Sin p(z=1): el Lasso no da probabilidades. Para la comparacion de activacion se usa |x_j|.
x_j_mean = l1_solution

print(f"n_src: {size}, seed: {seed}, alpha: {l1_alpha:.3e}, L1 norm diff: {l1_norm_diff:.4f}, "
      f"n_active: {int(l1_active.sum())}, time: {l1_time:.2f}s", flush=True)

# ---------- Guardar (un fichero por job) ----------
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
out_path = result_path(results_dir, size, seed)
print(f"Guardando resultados en {out_path}", flush=True)
np.savez_compressed(
    out_path,
    size=size, seed=seed, n_channels=n_channels,
    l1_mean=l1_solution, l1_norm_diff=l1_norm_diff, l1_time=l1_time, l1_alpha=l1_alpha,
    l1_active=l1_active, x_j_mean=x_j_mean,
)
