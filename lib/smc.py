import contextlib, io
from time import time
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import random
from lib.hmc import HMC, get_inverse_mass_matrix

def reweight(log_p_t, theta, itt):
    """ Computes the weights for the SMC algorithm based on the intermediate distribution p_t and the current samples theta and normalizes them.
    
    Arguments:
    p_t     -- Function implementing the logarithm of the intermediate distribution (up to an additive constant) at iteration itt. The function must return scalar values and be compatible with theta
    theta   -- Current samples, JaX array of shape NxD, where N is the number of samples and D is the dimension
    itt     -- Current iteration index

    Returns:
    weights -- Normalized weights for the current samples, JaX array of shape (N,)
    """
    log_weights = log_p_t(theta, itt) - log_p_t(theta, itt-1)  # Evaluate log of intermediate distribution at current samples
    weights = jnp.exp(log_weights - logsumexp(log_weights))  # Normalize the weights to sum to 1
    return weights

def SMC(log_prior, log_likelihood_t, log_target_ini, theta0, num_iterations_smc, num_iterations_hmc, S=10, num_leapfrog_steps_ini=10, num_leapfrog_steps_smc=10, step_size=1e-1, seed=0):
    """ Implements the basic SMC algorithm for D-dimensional target distribution specified by log_target
    
    Arguments:
    log_prior               -- Function implementing the logarithm of the prior distribution (up to an additive constant). The function must return scalar values and be compatible with theta0
    log_likelihood_t        -- Function implementing the logarithm of the likelihood for a given time t. The function must return scalar values and be compatible with theta0
    log_target_ini          -- Function implementing the logarithm of the likelihood to obtain mass matrix. The function must return scalar values and be compatible with theta0
    theta0                  -- Initial position, JaX array of shape 1xD, where D is the dimension
    num_iterations_smc      -- Number of SMC iterations
    num_iterations_hmc      -- Number of HMC iterations
    S                       -- Number of steps in muatation step
    num_leapfrog_steps_ini  -- Number of leapfrog steps for the initial stage
    num_leapfrog_steps_smc  -- Number of leapfrog steps for the SMC stage
    step_size               -- Step-size eta for leapfrog integration
    seed                    -- seed
    verbose                 -- Whether to print progress information
    
    Returns:    
    thetas                  -- MCMC samples, JaX array of shape (num_iterations+1, D)
    """
    N = theta0.shape[0]  # Number of particles
    key = random.PRNGKey(seed)
    theta, thetas = theta0, [theta0]   # list of SMC samples
    
    Sigma_est = get_inverse_mass_matrix(log_target_ini, theta0[0:1], num_iterations_hmc, num_leapfrog_steps_ini, step_size, seed)
    
    weights = jnp.ones(N) / N  # Initialize weights uniformly
    
    for itt in range(1, num_iterations_smc + 1):     
        time_start = time()   
        # Resampling step
        key, key_resample = random.split(key)
        indices = random.choice(key_resample, a=N, shape=(N,), p=weights)
        # For indices keep thetas, for the rest 
        theta = theta[indices]
    
        # Mutation step using HMC
        log_target_t = lambda theta: log_prior(theta) + log_likelihood_t(theta, itt)
        new = []
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(N):
                theta_i = theta[i:i+1]  # Current particle
                chain = HMC(log_target_t, S, theta_i, num_leapfrog_steps_smc, step_size*10000, seed=seed + itt * N + i, Sigma=Sigma_est)  # Take the last sample from HMC as the new theta
                new.append(chain[-1])  # Append the last sample of the chain to new particles
        theta = jnp.vstack(new)  # Stack the new particles into a single array
        thetas.append(theta)
        
        # Reweighing step
        weights = reweight(log_likelihood_t, theta, itt)
        
        print(f"SMC iteration {itt}/{num_iterations_smc} completed. Effective sample size: {1/jnp.sum(weights**2):.2f}, Time: {time() - time_start:.2f}s", flush=True)
            
    return thetas