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
from lib.hmc import HMC
from lib.map_solution import get_map_estimate

slab_var, spike_var, slab_prob = 0.1, 0.001, 0.5
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
sizes = sorted(set(range(1, n_sources + 1, 10)) | {n_channels})

# Si se ejecuta sin argumentos, solo informa de cuántos jobs hacen falta
if len(sys.argv) < 2:
    print(f"n_channels={n_channels}, n_sources={n_sources}, n_jobs={len(sizes)}  ->  #BSUB -J mapvshmc[1-{len(sizes)}]")
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

# ---------- MAP ----------
t0 = time()
map_solution = get_map_estimate(post.log_posterior, theta0=jnp.zeros(size))
map_time = time() - t0
map_solution = np.asarray(map_solution).ravel()
map_norm_diff = float(np.linalg.norm(map_solution - x_small[:, tid]))

# ---------- HMC ----------
t0 = time()
theta_map = jnp.expand_dims(map_solution, axis=0)
thetas = HMC(log_target=post.log_posterior, num_iterations=2000, theta0=theta_map,
             num_leapfrog_steps=250, step_size=1e-5, seed=0)
hmc_time = time() - t0
hmc_mean = jnp.mean(thetas[100:], axis=0)
hmc_norm_diff = float(jnp.linalg.norm(hmc_mean - x_small[:, tid]))

print(f"n_src: {size}, MAP norm diff: {map_norm_diff:.4f}, HMC norm diff: {hmc_norm_diff:.4f}", flush=True)

# ---------- Guardar (un fichero por job) ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Guardando resultados en {script_dir}/results/size_{size:04d}.npz", flush=True)
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)
np.savez(
    f"experiments/map_vs_hmc/results/size_{size:04d}.npz",
    size=size, n_channels=n_channels,
    map_mean=map_solution, map_norm_diff=map_norm_diff, map_time=map_time,
    hmc_mean=np.asarray(hmc_mean), hmc_norm_diff=hmc_norm_diff, hmc_time=hmc_time,
    thetas=np.asarray(thetas),
)