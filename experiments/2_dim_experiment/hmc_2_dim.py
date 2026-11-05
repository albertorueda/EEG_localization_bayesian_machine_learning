import sys, os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time
jax.config.update("jax_enable_x64", True)

from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from lib.distributions import SpikeAndSlabPrior, Posterior
from lib.hmc import HMC_inverse_mass
from lib.map_solution import get_map_estimate

slab_var, spike_var, slab_prob = 0.1, 0.001, 0.5
SpikeAndSlab = SpikeAndSlabPrior(slab_var=slab_var, spike_var=spike_var, slab_prob=slab_prob)

# ---------- TOY forward model: 1 sensor, 2 sources ----------
# One measurement cannot tell the two sources apart: the likelihood is a ridge
#   l0*theta0 + l1*theta1 = y
# and the spike-and-slab prior prefers points where one coordinate is ~0, so the
# posterior concentrates on two "arms": (y/l0, 0) and (0, y/l1).
L_toy = np.array([[1.0, 0.8],
                  [0.8, 1.0]])        # (n_sensors=1, n_sources=2), distinct columns
x_true = np.array([0.5, 0.0])          # only source 0 active
beta = 10.0                            # likelihood sharp enough to kill the (0,0) region

y = L_toy @ x_true                     # noise-free data (shape (1,)) -> clean picture
L_small = jnp.array(L_toy)
Y = jnp.array(y)[:, None]
y_tid = jnp.array(y)
size = 2
post = Posterior(prior_distribution=SpikeAndSlab, L=L_small, Y=Y, beta=beta, y_tid=y_tid)

# ---------- MAP ----------
t0 = time()
map_solution = get_map_estimate(post.log_posterior, theta0=jnp.zeros(size))
map_time = time() - t0
map_solution = np.asarray(map_solution).ravel()
map_norm_diff = float(np.linalg.norm(map_solution - x_true))
print(f"MAP: {map_solution}", flush=True)

# ---------- HMC ----------
# Stability: eps < 2/sqrt(lambda_max), lambda_max ~ beta^2*||l||^2 + 1/spike_var ~ 1.2e3  ->  eps < ~0.057
# Trajectory length eps*L ~ 1.5 is enough to cross between the two arms.
num_iterations = 2000
theta_init = jnp.expand_dims(jnp.asarray(map_solution), axis=0)
t0 = time()
thetas = HMC_inverse_mass(log_target=post.log_posterior, num_iterations=num_iterations, theta0=theta_init,
             num_leapfrog_steps=50, step_size=0.03, seed=0)
hmc_time = time() - t0
thetas_np = np.asarray(thetas)
hmc_mean = thetas_np[num_iterations//5:].mean(axis=0)
hmc_norm_diff = float(np.linalg.norm(hmc_mean - x_true))
print(f"n_src: {size}, MAP norm diff: {map_norm_diff:.4f}, HMC norm diff: {hmc_norm_diff:.4f}", flush=True)

# ---------- How often does the chain jump between arms? ----------
post_s = thetas_np[num_iterations//5:]
on_arm_A = np.abs(post_s[:, 1]) < 0.1        # theta1 ~ 0  (source 0 explains the data)
on_arm_B = np.abs(post_s[:, 0]) < 0.1        # theta0 ~ 0  (source 1 explains the data)
switches = int(np.sum(np.diff(on_arm_A.astype(int)) != 0))
print(f"frac arm A: {on_arm_A.mean():.2f}, frac arm B: {on_arm_B.mean():.2f}, arm switches: {switches}", flush=True)

# ---------- Plot: log-posterior contours + samples ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

g0 = np.linspace(-0.4, 1.0, 300)
g1 = np.linspace(-0.4, 1.0, 300)
G0, G1 = np.meshgrid(g0, g1)
grid = jnp.asarray(np.stack([G0.ravel(), G1.ravel()], axis=1))
logp = np.asarray(post.log_posterior(grid)).reshape(G0.shape)
logp = np.maximum(logp, logp.max() - 40)      # clip for readable contours

ax = axes[0]
cs = ax.contourf(G0, G1, logp, levels=40, cmap="viridis")
fig.colorbar(cs, ax=ax, label="log posterior")
ax.plot(*x_true, "r*", ms=14, label="true value")
ax.plot(*map_solution, "wx", ms=10, mew=2, label="MAP")
ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")
ax.set_title("Log posterior (2D)")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.contour(G0, G1, logp, levels=15, colors="gray", linewidths=0.5)
ax.plot(post_s[:, 0], post_s[:, 1], ".", ms=2, alpha=0.4, color="tab:blue")
ax.plot(*x_true, "r*", ms=14)
ax.set_xlabel(r"$\theta_0$"); ax.set_ylabel(r"$\theta_1$")
ax.set_title(rf"HMC samples (arm switches: {switches})")

ax = axes[2]
ax.plot(thetas_np[:, 0], lw=0.7, label=r"$\theta_0$")
ax.plot(thetas_np[:, 1], lw=0.7, label=r"$\theta_1$")
ax.axvline(num_iterations//5, color="r", ls=":", lw=1, label="burn-in")
ax.set_xlabel("HMC iteration"); ax.set_ylabel(r"$\theta$")
ax.set_title("Traces")
ax.legend(fontsize=8)

fig.tight_layout()

# ---------- Guardar ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)
fig.savefig(os.path.join(results_dir, "modes_2d.png"), dpi=150)
print(f"Guardando resultados en {results_dir}/size_{size:04d}.npz", flush=True)
np.savez(
    os.path.join(results_dir, f"size_{size:04d}.npz"),
    size=size, n_channels=L_toy.shape[0],
    map_mean=map_solution, map_norm_diff=map_norm_diff, map_time=map_time,
    hmc_mean=hmc_mean, hmc_norm_diff=hmc_norm_diff, hmc_time=hmc_time,
    thetas=thetas_np,
)