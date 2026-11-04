from jax import random
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class PriorDistribution:
    def log_prior(self, theta):
        """ Computes the log-prior of the distribution for a given parameter vector theta."""
        raise NotImplementedError("log_prior method must be implemented in subclasses.")

    def sample_prior(self, key, num_samples, num_features):
        """ Samples from the prior distribution."""
        raise NotImplementedError("sample_prior method must be implemented in subclasses.")

class tStudentPrior(PriorDistribution):
    def __init__(self, df, scale):
        self.df = df
        self.scale = scale
        
    def log_prior(self, theta):
        """ Computes the log-prior of the t-Student distribution for a given parameter vector theta."""
        return -0.5 * (self.df + 1) * jnp.sum(jnp.log1p(theta**2 / (self.df * self.scale**2)), axis=-1)
    
    def sample_prior(self, key, num_samples, num_features):
        """ Samples from the t-Student prior distribution."""
        return random.t(key, df=self.df, shape=(num_samples, num_features)) * self.scale

class SpikeAndSlabPrior(PriorDistribution):
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

class HorseShoePrior(PriorDistribution):
    def __init__(self, tau):
        self.tau = tau

    def log_prior(self, theta):
        """ Computes the log-prior of the horseshoe distribution for a given theta """
        pass
    
    def sample_prior(self, key, num_samples, num_features):
        """ Samples from the horseshoe prior distribution """
        pass


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