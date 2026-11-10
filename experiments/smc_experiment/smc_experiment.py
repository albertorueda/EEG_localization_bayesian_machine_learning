from jax import random
import sys, os
import jax
import jax.numpy as jnp
import numpy as np
import mne
from time import time
jax.config.update("jax_enable_x64", True)

from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from lib.distributions import SpikeAndSlabPrior, Posterior
from lib.smc import SMC
from lib.map_solution import get_map_estimate

slab_var, spike_var, slab_prob = 0.1, 0.001, 0.05
SpikeAndSlab = SpikeAndSlabPrior(slab_var=slab_var, spike_var=spike_var, slab_prob=slab_prob)

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

# `size` = nº de FUENTES (columnas de L). Siempre se usan TODOS los sensores.
# Incluimos n_channels para ver bien la transición sobredeterminado -> infradeterminado.
sizes = sorted(set(range(5, 301, 5)) | {300, 1})

# Si se ejecuta sin argumentos, solo informa de cuántos jobs hacen falta
if len(sys.argv) < 2:
    print(f"n_channels={n_channels}, n_sources={n_sources}, n_jobs={len(sizes)}  ->  #BSUB -J smc[1-{len(sizes)}]")
    sys.exit(0)

job_idx = int(sys.argv[1])          # $LSB_JOBINDEX (1-based)
size = sizes[job_idx - 1]
print(f"Job {job_idx}: n_channels={n_channels}, n_src(size)={size}", flush=True)

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

# ---------- SMC ----------
N = 1000                             # tiene que ser > size
theta0 = SpikeAndSlab.sample_prior(random.PRNGKey(0), num_samples=N, num_features=size)
 
t0 = time()
smc_particles, hist = SMC(
    log_prior=SpikeAndSlab.log_prior,
    log_lik=post.log_likelihood,     # verosimilitud completa; el SMC aplica lambda
    theta0=theta0,
    seed=0,
    step_size=0.25, n_leapfrog=10, n_mcmc=5, ess_frac=0.5,
)
smc_time = time() - t0
 
smc_param = smc_particles.mean(axis=0)           # (size,) media posterior
smc_norm_diff = float(jnp.linalg.norm(smc_param - x_small[:, tid]))
 
print(f"n_src: {size}, SMC norm diff: {smc_norm_diff:.4f}, time: {smc_time:.1f}s", flush=True)
 
# ---------- Guardar ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Guardando resultados en {script_dir}/results/size_{size:04d}.npz", flush=True)
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)
np.savez(
    f"experiments/smc_experiment/results/size_{size:04d}.npz",
    size=size, n_channels=n_channels,
    smc_mean=np.asarray(smc_param), smc_norm_diff=smc_norm_diff, smc_time=smc_time,
    thetas=np.asarray(smc_particles),
    lam=np.asarray(hist["lam"]), ess=np.asarray(hist["ess"]),
    acc=np.asarray(hist["acc"]), logZ=np.asarray(hist["logZ"]),
)