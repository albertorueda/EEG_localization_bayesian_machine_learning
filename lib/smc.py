"""
smc.py -- SMC sampler with adaptive likelihood tempering, following Murphy, "Probabilistic Machine Learning:
Advanced Topics", chapter 13.

Book pieces used
  * Algorithm 13.5 (SMC with adaptive tempering) on the geometric path (13.75-13.76):
        p_lambda(theta) ∝ prior(theta) * likelihood(theta)^lambda,   lambda: 0 -> 1
    Particles start from the prior (lambda = 0).
  * Incremental weights exp(-delta * E(theta)) with E = -log likelihood (13.77), evaluated at the particles
    BEFORE the MCMC move, as in (13.74).
  * delta_t = argmin over [0, 1 - lambda] of (ESS({-delta * E}) - ESS_min)^2 (13.78), with ESS_min = ess_frac * N.
    The book writes the difference without the square; the square makes the minimum sit where ESS = ESS_min.
    If delta = 1 - lambda already gives ESS >= ESS_min there is no solution in the interval and, as in the book,
    delta = 1 - lambda. The minimisation is done over log10(delta) (scipy, bounded Brent) because delta can be
    anywhere between 1e-9 and 1.
  * ESS = 1 / sum(W^2)   (13.36).
  * Systematic resampling (13.2.4.4), recommended over multinomial in 13.2.4.5.
  * Forward kernel M_t = MCMC kernel invariant for p_lambda, here several HMC steps (13.6.1: "in high dimensional
    problems ... gradient based kernels such as HMC"; 13.5 "resample-move").
  * Inverse mass matrix = empirical covariance (12.5.4.3). NOTE: 13.6.1 gives the weighted covariance of the previous
    step only for the RWMH proposal (delta^2 * Sigma_hat); using it as the HMC mass matrix is a combination of
    12.5.4.3 and 13.6.1, not something the chapter states for HMC.
  * Normalizing constant: Z_t / Z_{t-1} ≈ mean of the unnormalized incremental weights (13.26-13.28).

Implementation only (no change to the algorithm)
  * All N particles are mutated in a single jitted call instead of N separate HMC runs.
  * The HMC step size `step_size`, n_leapfrog and n_mcmc are fixed inputs (the book gives no rule for them). The acceptance
    rate of every stage is stored in hist["acc"] so you can check step_size.

Remarks
  * The empirical covariance has rank <= N - 1, so use N > D (a warning is printed otherwise). Only a tiny jitter is
    added to the diagonal.
  * Both functions you pass in must accept a batch of shape (N, D):
        log_prior(theta) -> (N,)
        log_lik(theta)   -> (N,)     e.g. Posterior.log_likelihood
"""
from functools import partial
from time import time

import jax.numpy as jnp
from jax import grad, jit, lax, random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------- helpers
@jit
def ess_from_logw(logw):
    """ESS = 1 / sum(W^2) from unnormalized log weights (13.36)."""
    w = jnp.exp(logw - logsumexp(logw))
    return 1.0 / jnp.sum(w ** 2)


def next_delta(log_lik, lam, ess_min, t_min=-16.0):
    """delta_t of (13.78): argmin over delta in [0, 1 - lam] of (ESS({-delta * E}) - ESS_min)^2, E = -log_lik.
 
    Searched over t = log10(delta) in [t_min, log10(1 - lam)]: the objective is flat when delta is far from the
    solution (ESS ~ 1 for large delta, ESS ~ N for tiny delta), and a search in log scale keeps the minimum
    reachable when the likelihood is very sharp and delta is of order 1e-6 or smaller.
    """
    if float(ess_from_logw((1.0 - lam) * log_lik)) >= ess_min:      # no solution in the interval
        return 1.0 - lam
    f = lambda t: (float(ess_from_logw((10.0 ** t) * log_lik)) - ess_min) ** 2
    res = minimize_scalar(f, bounds=(t_min, float(np.log10(1.0 - lam))), method="bounded",
                          options={"xatol": 1e-12, "maxiter": 500})
    return float(10.0 ** res.x)


def systematic_resample(key, W):
    """Ancestor indices by systematic resampling (13.35) through the inverse cdf (13.33)."""
    N = W.shape[0]
    u = (random.uniform(key) + jnp.arange(N)) / N # uniformly spaced points in [0, 1)
    return jnp.minimum(jnp.searchsorted(jnp.cumsum(W), u), N - 1) # searchsorted returns N if u[-1] = 1.0, so clip to N - 1 (minimum so it works with doesnt fail in case of rounding errors)


def weighted_mean_cov(theta, W, jitter=1e-10):
    """Weighted mean and covariance of the particles: Sigma = E[(theta - mean)(theta - mean)^T], the inverse mass
       matrix of 12.5.4.3, computed from the weighted particles instead of from a warm-up run."""
    mu = W @ theta
    S = jnp.cov(theta.T, aweights=W, bias=True)
    return mu, S + jitter * jnp.eye(theta.shape[1])

# ---------------------------------------------------------------- batched HMC kernel
def make_kernel(log_target):
    """log_target(theta (N, D), lam) -> (N,). Returns a jitted batched HMC step.

    HMC runs in whitened coordinates phi, theta = mu + phi @ A.T with A A^T = Sigma (inverse mass matrix), so the
    kinetic energy is 0.5*|nu|^2 and no matrix inverse is needed.
    """

    @partial(jit, static_argnames=("n_leapfrog",))
    def step(key, theta, mu, A, lam, step_size, n_leapfrog):
        phi = solve_triangular(A, (theta - mu).T, lower=True).T # whitened coordinates, phi = A^{-1} (theta - mu)

        def U(p):                                    # potential energy of each particle, (N,)
            return -log_target(mu + p @ A.T, lam)

        gradU = grad(lambda p: jnp.sum(U(p)))        # particles are independent -> per-particle gradient

        key_nu, key_u = random.split(key)
        nu = random.normal(key_nu, phi.shape)
        H0 = U(phi) + 0.5 * jnp.sum(nu ** 2, axis=1)

        def body(_, s):
            p, v, g = s
            v = v - 0.5 * step_size * g
            p = p + step_size * v
            g = gradU(p)
            v = v - 0.5 * step_size * g
            return p, v, g

        p1, v1, _ = lax.fori_loop(0, n_leapfrog, body, (phi, nu, gradU(phi)))
        log_acc = H0 - (U(p1) + 0.5 * jnp.sum(v1 ** 2, axis=1))
        log_acc = jnp.where(jnp.isfinite(log_acc), log_acc, -jnp.inf)
        accept = jnp.log(random.uniform(key_u, (phi.shape[0],))) < log_acc
        phi_new = jnp.where(accept[:, None], p1, phi)
        return mu + phi_new @ A.T, jnp.mean(jnp.exp(jnp.minimum(log_acc, 0.0)))

    return step


# ---------------------------------------------------------------- Algorithm 13.5
def SMC(log_prior, log_lik, theta0, seed=0, ess_frac=0.5, n_mcmc=5, n_leapfrog=10,
        step_size=0.1, max_stages=500, verbose=True):
    """
    theta0 : (N, D) particles drawn from the prior (Alg. 13.5, line 5).
    step_size : fixed HMC step size, in whitened coordinates (check hist["acc"]).
    Returns (theta, hist): equally weighted particles (N, D) approximating the posterior, and a dict with, per stage,
    lambda, ESS, acceptance and the running estimate log Z of the marginal likelihood.
    """
    N, D = theta0.shape
    if N <= D:
        print(f"WARNING: N={N} <= D={D}: the particle covariance is singular; use N > D.", flush=True)
    key = random.PRNGKey(seed)
    ll_fn = jit(log_lik)
    kernel = make_kernel(lambda th, lam: log_prior(th) + lam * log_lik(th))

    theta, lam = theta0, 0.0
    hist = dict(lam=[0.0], ess=[], acc=[], logZ=[0.0])
    t_start = time()

    for stage in range(1, max_stages + 1):
        # lines 9-12: delta (13.78), lambda_t, incremental weights exp(-delta E) at the current particles
        ll = ll_fn(theta)
        delta = next_delta(ll, lam, ess_frac * N)
        logw = delta * ll
        W = jnp.exp(logw - logsumexp(logw))
        ess = float(ess_from_logw(logw))
        hist["logZ"].append(hist["logZ"][-1] + float(logsumexp(logw) - jnp.log(N)))   # (13.27-13.28)
        lam_new = min(1.0, lam + delta)

        # inverse mass matrix Sigma = empirical covariance of the weighted particles. Section 12.5.4.3 estimates Sigma
        # from warm-up HMC samples; here the warm-up is replaced by the weighted particles of the previous step
        # (the weighted covariance of 13.6.1, which the book gives for RWMH).
        mu, S = weighted_mean_cov(theta, W)
        A = jnp.linalg.cholesky(S)

        # lines 7-8: resample, then move with a kernel invariant for p_lambda_new
        key, k_res = random.split(key)
        theta = theta[systematic_resample(k_res, W)]
        lam = lam_new
        accs = []
        for _ in range(n_mcmc):
            key, k_mut = random.split(key)
            theta, acc = kernel(k_mut, theta, mu, A, lam, step_size, n_leapfrog)
            accs.append(float(acc))

        hist["lam"].append(lam)
        hist["ess"].append(ess)
        hist["acc"].append(sum(accs) / len(accs))
        if verbose:
            print(f"stage {stage:3d}  lambda={lam:.5f}  ESS={ess:7.1f}/{N}  "
                  f"acc={hist['acc'][-1]:.2f}  logZ={hist['logZ'][-1]:.2f}  "
                  f"t={time() - t_start:.1f}s", flush=True)
        if lam >= 1.0:
            break

    return theta, hist