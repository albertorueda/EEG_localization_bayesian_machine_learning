import jax.numpy as jnp
from jax import random, grad, jit
from time import time
from jax.scipy.stats import multivariate_normal as mvn


def build_energy_functions(log_target, Sigma=None):
    def kinetic_energy(nu):
        if Sigma is None:
            return 0.5 * jnp.sum(nu**2, axis=-1) if nu.ndim > 1 else 0.5 * jnp.sum(nu**2)
        else:
            return 0.5 * jnp.sum((nu@Sigma)*nu, axis=-1)
    
    def potential_energy(theta):
        return -log_target(theta).squeeze()
    
    def hamiltonian(theta, nu):
        return potential_energy(theta) + kinetic_energy(nu)
    
    return hamiltonian, potential_energy, kinetic_energy, grad(potential_energy)

def leapfrog(theta0, nu0, grad_pot_energy_fun, num_steps, step_size, Sigma=None, full_trajectory=True):
    if full_trajectory:
        theta_trajectory = [theta0]
        nus_trajectory = [nu0]
    
    theta, nu = theta0, nu0
    for k in range(num_steps):
        nu = nu - 0.5 * step_size * grad_pot_energy_fun(theta)
        
        if Sigma is None:
            theta = theta + step_size * nu
        else:
            # Multiplicación elemento a elemento si Sigma es diagonal
            theta = theta + step_size * (nu@Sigma)
            
        nu = nu - 0.5 * step_size * grad_pot_energy_fun(theta)

        if full_trajectory:
            theta_trajectory.append(theta)
            nus_trajectory.append(nu)
            
    if full_trajectory:
        return theta, nu, jnp.vstack(theta_trajectory), jnp.vstack(nus_trajectory)
    else:
        return theta, nu
    
def HMC(log_target, num_iterations, theta0, num_leapfrog_steps=10, step_size=1e-1, Sigma=None, seed=0, use_jit=True):
    hamiltonian, potential_energy, kinetic_energy, grad_pot_energy_fun = build_energy_functions(log_target, Sigma=Sigma)
    key = random.PRNGKey(seed)

    param_shape = theta0.shape
    theta = theta0
    thetas = [theta0]
    accept_count = 0
    t0 = time()

    if use_jit:
        hamiltonian = jit(hamiltonian)
        grad_pot_energy_fun = jit(grad_pot_energy_fun)

    for itt in range(num_iterations):        
        key, key_momentum, key_accept = random.split(key, 3)

        # Muestreo de momento ajustado por Sigma: nu ~ N(0, Sigma^{-1})
        # Si Sigma es diagonal, la desviación estándar de nu es 1 / sqrt(Sigma)
        if Sigma is None:
            nu = random.normal(key_momentum, shape=param_shape)
        else:
            nu = random.multivariate_normal(key_momentum, mean=jnp.zeros(param_shape), cov=jnp.linalg.inv(Sigma))

        theta_star, nu_star = leapfrog(theta, nu, grad_pot_energy_fun, num_leapfrog_steps, step_size, Sigma=Sigma, full_trajectory=False)

        log_accept_prob = -hamiltonian(theta_star, nu_star) + hamiltonian(theta, nu)
    
        u = random.uniform(key_accept)
        if jnp.log(u) < log_accept_prob:
            theta = theta_star
            accept_count = accept_count + 1
 
        thetas.append(theta)

        if (itt+1) % max(1, int(num_iterations/10)) == 0:
            print(f'Itt {itt+1:5d}/{num_iterations} ({100*(itt+1)/num_iterations:6.2f}%), accept_rate={accept_count/(itt+1):.3f}, time={time()-t0:3.2f}s')

    print(f'HMC done in {time()-t0:.3f}s with an avg. accept rate={accept_count/num_iterations:.3f}\n\n')

    return jnp.vstack(thetas)

def get_inverse_mass_matrix(log_target, theta0, num_iterations=200, num_leapfrog_steps=150, step_size=1e-1, seed=0):
    thetas_warmup = HMC(log_target, num_iterations, theta0, num_leapfrog_steps=num_leapfrog_steps, step_size=step_size, seed=seed)
    Sigma_est = jnp.cov(thetas_warmup.T) + 1e-6 * jnp.eye(thetas_warmup.shape[1])  # Add a small value to the diagonal for numerical stability
    return Sigma_est

def HMC_inverse_mass(log_target, num_iterations, theta0, num_leapfrog_steps=10, step_size=1e-1, seed=0):
    Sigma_est = get_inverse_mass_matrix(log_target, theta0, num_iterations//5, num_leapfrog_steps=num_leapfrog_steps, step_size=step_size, seed=seed)
    
    thetas = HMC(log_target, num_iterations, theta0, num_leapfrog_steps=num_leapfrog_steps//5, step_size=10000*step_size, Sigma=Sigma_est, seed=seed)
    return thetas