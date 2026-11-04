import numpy as np
from jax import numpy as jnp
from lib.utils import get_lead_field_matrix
from lib.distributions import SpikeAndSlabPrior, Posterior
from lib.map_solution import get_map_estimate
from lib.hmc import HMC
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

sfreq = 250
L = get_lead_field_matrix(sfreq=sfreq)

t = np.linspace(0, 1, sfreq, endpoint=False)
A = 0.5
x_1 = A * np.sin(2 * np.pi * t)
X = np.zeros((L.shape[1], sfreq))
X[0] = x_1
beta = 4
e = np.random.normal(0, 1/beta, size=(L.shape[0], sfreq))
Y = L @ X + e

tid = 62
size=1000
L_small = jnp.array(L[0:size, 0:size])
x_small = X[0:size]
Y = L_small @ x_small + e[0:size, :]
y_tid = Y[:, tid]
slab_var, spike_var, slab_prob = 0.1, 0.001, 0.5
spike_and_slab_prior = SpikeAndSlabPrior(slab_var=slab_var, spike_var=spike_var, slab_prob=slab_prob)
spike_and_slab_posterior = Posterior(prior_distribution=spike_and_slab_prior, L=L_small, Y=Y, beta=beta, y_tid=y_tid)

theta0 = jnp.zeros(L_small.shape[1])
theta_map = get_map_estimate(spike_and_slab_posterior.log_posterior, theta0=theta0)
theta_map = jnp.expand_dims(theta_map, axis=0)
thetas = HMC(spike_and_slab_posterior.log_posterior, num_iterations=2000, theta0=theta0, num_leapfrog_steps=250, step_size=1e-5, seed=0)
fig = plt.figure(figsize=(15, 4))
plt.plot(np.linspace(0, 1000, thetas.shape[0]), thetas[:, 0], label='Trace of x_0', color='blue')
plt.plot(np.linspace(0, 1000, thetas.shape[0]), thetas[:, 1:], label='Trace of x_1', color='orange')
plt.savefig('hmc_trace_x1.png')
