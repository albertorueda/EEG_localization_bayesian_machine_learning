import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def get_map_estimate(log_posterior, theta0, method='BFGS'):
    """
    Compute the MAP estimate using optimization.

    Parameters:
    - log_posterior: Function to compute the log-posterior.
    - theta0: Initial guess for the parameters.
    - method: Optimization method (default is 'BFGS').

    Returns:
    - result.x: The MAP estimate of the parameters.
    """
    def objective(theta):
        return float(
            -log_posterior(
                jnp.asarray(theta)
            )
        )

    def gradient(theta):
        grad_fn = jax.grad(
            lambda t: -log_posterior(t)
        )
        return np.asarray(
            grad_fn(jnp.asarray(theta)), dtype=float
        )

    result = minimize(
        objective,
        x0=theta0,
        jac=gradient,
        method="L-BFGS-B"
    )
    return result.x