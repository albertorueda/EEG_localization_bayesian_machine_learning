import sys, os
import contextlib, io
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import mne
from jax.scipy.special import logsumexp
from jax import random
from jax import grad, jit
from time import time
jax.config.update("jax_enable_x64", True)

results_HMC = []
results_MAP = []

class SpikeAndSlabPrior:
    def __init__(self, slab_var, spike_var, slab_prob):
        self.slab_var = slab_var
        self.spike_var = spike_var
        self.slab_prob = slab_prob

    def log_prior(self, theta):
        """ Computes the log-prior of the spike-and-slab distribution for a given theta """
        ls = jnp.log(self.slab_prob)     - 0.5*jnp.log(2*jnp.pi*self.slab_var)  - theta**2/(2*self.slab_var)
        lp = jnp.log(1 - self.slab_prob) - 0.5*jnp.log(2*jnp.pi*self.spike_var) - theta**2/(2*self.spike_var)
        return jnp.sum(logsumexp(jnp.stack([ls, lp]), axis=0), axis=-1)
    
    def sample_prior(self, key, num_samples, num_features):
        """ Samples from the spike-and-slab prior distribution """
        slab_key, spike_key, mask_key = random.split(key, 3)
        slab_samples = random.normal(slab_key, shape=(num_samples, num_features)) * jnp.sqrt(self.slab_var)
        spike_samples = random.normal(spike_key, shape=(num_samples, num_features)) * jnp.sqrt(self.spike_var)
        mask = random.bernoulli(mask_key, p=self.slab_prob, shape=(num_samples, num_features))
        return mask * slab_samples + (1 - mask) * spike_samples
    
class Posterior:
    def __init__(self, prior_distribution, L, Y, beta, y_tid):
        self.prior_distribution = prior_distribution
        self.L = L
        self.Y = Y
        self.beta = beta
        self.y_tid = y_tid

    def log_likelihood(self, theta):
        noise_prec = self.beta**2

        if theta.ndim == 1:  # MAP: shape (D,)
            residuals = self.y_tid - self.L @ theta
            return -0.5 * noise_prec * jnp.sum(residuals**2)

        # HMC: shape (N, D)
        residuals = self.y_tid[:, None] - self.L @ theta.T
        return -0.5 * noise_prec * jnp.sum(residuals**2, axis=0)

    def log_posterior(self, theta):
        return (
            self.prior_distribution.log_prior(theta)
            + self.log_likelihood(theta)
        )

    def log_likelihood_t(self, theta, t):
        noise_prec = self.beta**2

        if theta.ndim == 1:
            residuals = self.y_tid[:t] - self.L[:t, :] @ theta
            return -0.5 * noise_prec * jnp.sum(residuals**2)

        residuals = (
            self.y_tid[:t, None] - self.L[:t, :] @ theta.T
        )
        return -0.5 * noise_prec * jnp.sum(residuals**2, axis=0)
    

def build_energy_functions(log_target):
    """ Helper function for constructing the relevant energy functions and gradients for HMC based on the given log_target function.
        The function log_target must accept JaX arrays of shape NxD, where D is the dimension of the parameter space and N
        is the number of points to be evaluated.
    """

    def kinetic_energy(nu):
        """ evaluates kinetic energy for momentum nu (shape: NxD), returns (N) array of kinetic energies"""
        N, D = nu.shape
        const = 0.5*D*jnp.log(2*jnp.pi) # can be left out
        return const + 0.5*jnp.sum(nu**2, 1).squeeze()
    
    def potential_energy(theta):
        """ evaluates potential energy for position theta (shape: NxD), returns (N) array of potential energies """
        return -log_target(theta).squeeze()
    
    def hamiltonian(theta, nu):
        """ evaluates Hamiltonian for state (theta, nu), returns (N) array of Hamiltonian values """
        return potential_energy(theta) + kinetic_energy(nu)
    
    return hamiltonian, potential_energy, kinetic_energy, grad(potential_energy)


def leapfrog(theta0, nu0, grad_pot_energy_fun, num_steps, step_size, full_trajectory=True):
    """ computes hamiltonian dynamics using leapfrog integration for D-dimensional problem

        Arguments:
        theta0                  --      Initial position, shape 1xD
        nu0                     --      Initial momentum, shape 1xD
        grad_pot_energy_fun     --      Function for evaluating the gradient of the potential energy, must be compatible with theta0
        num_steps               --      Number of leapfrog steps
        step_size               --      Step-size for integration, positive scalar
        full_trajectory         --      Flag indicating whether the function should return the full trajectory or only the last state.

        Returns:
        theta                   --      Final position after num_steps leapfrog steps
        nu                      --      Final momentum after num_steps leapfrog steps
        theta_trajectory        --      Array of position values for each step (shape: (num_steps+1) x D) if full_trajectory is True
        nu_trajectory           --      Array of momentum values for each step (shape: (num_steps+1) x D) if full_trajectory is True
    """

    if full_trajectory:
        theta_trajectory = [theta0]
        nus_trajectory = [nu0]
    
    theta, nu = theta0, nu0
    for k in range(num_steps):
        nu = nu - 0.5*step_size*grad_pot_energy_fun(theta)
        theta = theta + step_size*nu
        nu = nu - 0.5*step_size*grad_pot_energy_fun(theta)

        if full_trajectory:
            theta_trajectory.append(theta)
            nus_trajectory.append(nu)
            
    if full_trajectory:
        return theta, nu, jnp.vstack(theta_trajectory), jnp.vstack(nus_trajectory)
    else:
        return theta, nu
    
    
def HMC(log_target, num_iterations, theta0, num_leapfrog_steps=10, step_size=1e-1, seed=0):
    """ Implements the basic HMC algorithm for D-dimensional target distribution specified by log_target
    
    Arguments:
    log_target          -- Function implementing the logarithm of the target distribution (up to an additive constant). The function must return scalar values and be compatible with theta0
    num_iterations      -- Number of iterations
    theta0              -- Initial position, JaX array of shape 1xD, where D is the dimension
    num_leapfrog_steps  -- Number of leapfrog steps
    step_size           -- Step-size eta for leapfrog integration
    seed                -- seed

    Returns:
    thetas              -- MCMC samples, JaX array of shape (num_iterations+1, D)
    """
    hamiltonian, potential_energy, kinetic_energy, grad_pot_energy_fun = build_energy_functions(log_target)
    key = random.PRNGKey(seed)

    param_shape = theta0.shape
    theta = theta0
    thetas = [theta0]   # list of MCMC samples
    accept_count = 0    # for counting number of accepted samples
    t0 = time()

    # use just-in-time (jit) compilation to speed up computations - comment out if debugging becomes tricky
    hamiltonian = jit(hamiltonian)
    grad_pot_energy_fun = jit(grad_pot_energy_fun)

    for itt in range(num_iterations):        
        # update keys
        key, key_momentum, key_accept = random.split(key, 3)

        # resample momentum
        nu = random.normal(key_momentum, shape=param_shape)

        # propose
        theta_star, nu_star = leapfrog(theta, nu, grad_pot_energy_fun, num_leapfrog_steps, step_size, full_trajectory=False)

        # Compute log acceptance probability
        log_accept_prob = -hamiltonian(theta_star, nu_star) + hamiltonian(theta, nu)
    
        # accept/reject
        u = random.uniform(key_accept)
        if jnp.log(u) < log_accept_prob:
            theta = theta_star
            accept_count = accept_count + 1
 
        thetas.append(theta)

        # verbose
        if (itt+1) % int(num_iterations/10) == 0:
            print(f'Itt {itt+1:5d}/{num_iterations} ({100*(itt+1)/num_iterations:6.2f}%), accept_rate={accept_count/(itt+1):.3f}, time={time()-t0:3.2f}s')

    print(f'HMC done in {time()-t0:.3f}s with an avg. accept rate={accept_count/num_iterations:.3f}\n\n')

    return jnp.vstack(thetas)

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

n_sources = L.shape[1]
sizes = list(range(2, n_sources + 1, 10))

# Si se ejecuta sin argumentos, solo informa de cuántos jobs hacen falta
if len(sys.argv) < 2:
    print(f"n_sources={n_sources}, n_jobs={len(sizes)}  ->  #BSUB -J mapvshmc[1-{len(sizes)}]")
    sys.exit(0)

job_idx = int(sys.argv[1])          # $LSB_JOBINDEX (1-based)
size = sizes[job_idx - 1]
print(f"Job {job_idx}: size={size}", flush=True)

# ---------- Datos sintéticos (misma semilla en TODOS los jobs) ----------
tid = 62
fs = int(info["sfreq"])
t = np.linspace(0, 1, fs, endpoint=False)
A = 0.5
X = np.zeros((L.shape[1], fs))
X[0] = A * np.sin(2 * np.pi * t)

beta = 4
rng = np.random.default_rng(12345)   # <- fija: el ruido es el mismo en todos los jobs
e = rng.normal(0, 1/beta, size=(L.shape[0], fs))

# ---------- Problema de tamaño `size` ----------
L_small = jnp.array(L[0:size, 0:size])
x_small = X[0:size]
Y = L_small @ x_small + e[0:size, :]
y_tid = Y[:, tid]
post = Posterior(prior_distribution=SpikeAndSlab, L=L_small, Y=Y, beta=beta, y_tid=y_tid)

# ---------- MAP ----------
grad_fn = jax.jit(jax.grad(lambda th: -post.log_posterior(th)))
obj_fn = jax.jit(lambda th: -post.log_posterior(th))

t0 = time()
result = minimize(
    lambda th: float(obj_fn(jnp.asarray(th))),
    x0=np.zeros(L_small.shape[1]),
    jac=lambda th: np.asarray(grad_fn(jnp.asarray(th)), dtype=float),
    method="L-BFGS-B",
)
map_time = time() - t0
map_norm_diff = float(np.linalg.norm(result.x - x_small[:, tid]))

# ---------- HMC ----------
theta0 = jnp.zeros((1, L_small.shape[1]))
t0 = time()
thetas = HMC(log_target=post.log_posterior, num_iterations=2000, theta0=theta0,
             num_leapfrog_steps=250, step_size=1e-5, seed=0)
hmc_time = time() - t0
hmc_mean = jnp.mean(thetas[100:], axis=0)
hmc_norm_diff = float(jnp.linalg.norm(hmc_mean - x_small[:, tid]))

print(f"Size: {size}, MAP norm diff: {map_norm_diff:.4f}, HMC norm diff: {hmc_norm_diff:.4f}", flush=True)

# ---------- Guardar (un fichero por job) ----------
os.makedirs("results", exist_ok=True)
np.savez(
    f"results/size_{size:04d}.npz",
    size=size,
    map_mean=result.x, map_norm_diff=map_norm_diff, map_time=map_time,
    hmc_mean=np.asarray(hmc_mean), hmc_norm_diff=hmc_norm_diff, hmc_time=hmc_time,
    thetas=np.asarray(thetas),
)