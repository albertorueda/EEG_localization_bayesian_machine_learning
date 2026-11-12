import jax.numpy as jnp
from jax import random, grad, jit, hessian
from time import time


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

def acceptance_rate(thetas):
    """Fraction of iterations in which the chain moved (= accepted proposals)."""
    th = jnp.asarray(thetas)
    return float(jnp.mean(jnp.any(jnp.diff(th, axis=0) != 0, axis=1)))
 
 
def initial_step_size(log_target, theta0, default=1e-2):
    """12.5.4.2 with Sigma = I: eta_0 ~ smallest marginal std ~ 1 / sqrt(lambda_max(Hessian of -log_target))."""
    f = lambda th: -log_target(th).squeeze()
    f_grad = grad(f)
    H = hessian(f)(theta0)
    lam_max = float(jnp.linalg.eigvalsh(0.5 * (H + H.T))[-1])
    return 1.0 / jnp.sqrt(lam_max) if jnp.isfinite(lam_max) and lam_max > 0 else default
 
 
def tune_step_size(log_target, theta0, Sigma=None, eta0=1e-2, n_iter=100, lo=0.4, hi=0.8,
                   max_tries=40, seed=0):
    """Heuristic of [BZ20] quoted in 12.5.4.2: L = 1, vary eta until acceptance is in [lo, hi].
    Too low acceptance -> eta / 2; too high -> eta * 1.5. Returns (eta, acceptance)."""
    eta, best, acc = eta0, None, jnp.nan
    for _ in range(max_tries):
        th = HMC(log_target, n_iter, theta0, num_leapfrog_steps=1, step_size=eta, Sigma=Sigma, seed=seed)
        acc = acceptance_rate(th)
        if acc >= lo:
            best = eta if best is None else max(best, eta)
        if lo <= acc <= hi:
            return eta, acc
        eta *= 0.5 if acc < lo else 1.5
    return (best if best is not None else eta), acc
 
 
def HMC_inverse_mass(log_target, num_iterations, theta0, num_leapfrog_steps=10, num_warmup=1000,
                           seed=0, verbose=True):
    """theta0: (1, D). Returns thetas (num_iterations + 1, D), like HMC()."""
    D = theta0.shape[-1]
 
    # 1-2) eta with Sigma = I, L = 1
    eta0 = initial_step_size(log_target, theta0)
    eta_I, acc_I = tune_step_size(log_target, theta0, None, eta0, seed=seed)
 
    # 3) warm-up with Sigma = I (moves, because eta is tuned)
    warm = jnp.asarray(HMC(log_target, num_warmup, theta0, num_leapfrog_steps=num_leapfrog_steps,
                                 step_size=eta_I, seed=seed))
    w = warm[num_warmup // 2:]                                   # drop first half
 
    # 4) Sigma = covariance of warm-up samples (12.5.4.3); tiny relative jitter for numerical stability
    S = jnp.atleast_2d(jnp.cov(w.T))
    S = S + 1e-6 * jnp.mean(jnp.diag(S)) * jnp.eye(D)
    Sigma = jnp.asarray(S)
 
    # 5) re-tune eta in the whitened geometry (not in the book)
    theta_start = jnp.asarray(warm[-1:])                          # (1, D)
    eta_S, acc_S = tune_step_size(log_target, theta_start, Sigma, 0.1, seed=seed)
 
    if verbose:
        print(f"[tuned HMC] D={D}  eta0={eta0:.2e}  eta(Sigma=I)={eta_I:.2e} (acc {acc_I:.2f})  "
              f"warm-up acc={acceptance_rate(warm):.2f}  eta(Sigma)={eta_S:.2e} (acc {acc_S:.2f})", flush=True)
 
    # 6) final chain
    return HMC(log_target, num_iterations, theta_start, num_leapfrog_steps=num_leapfrog_steps,
               step_size=eta_S, Sigma=Sigma, seed=seed)