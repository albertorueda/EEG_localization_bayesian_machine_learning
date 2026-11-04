import jax.numpy as jnp
from jax import random, grad, jit
from time import time


def build_energy_functions(log_target):
    """ Helper function for constructing the relevant energy functions and gradients for HMC based on the given log_target function.
        The function log_target must accept JaX arrays of shape NxD, where D is the dimension of the parameter space and N
        is the number of points to be evaluated.
    """

    def kinetic_energy(nu):
        """Evaluate kinetic energy for one or more momentum vectors."""
        if nu.ndim == 1:
            return 0.5 * jnp.sum(nu**2)

        D = nu.shape[-1]
        const = 0.5 * D * jnp.log(2 * jnp.pi)
        return const + 0.5 * jnp.sum(nu**2, axis=-1)
    
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
        if (itt+1) % max(1, int(num_iterations/10)) == 0:
            print(f'Itt {itt+1:5d}/{num_iterations} ({100*(itt+1)/num_iterations:6.2f}%), accept_rate={accept_count/(itt+1):.3f}, time={time()-t0:3.2f}s')

    print(f'HMC done in {time()-t0:.3f}s with an avg. accept rate={accept_count/num_iterations:.3f}\n\n')

    return jnp.vstack(thetas)