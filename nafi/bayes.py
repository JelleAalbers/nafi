from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import nafi

export, __all__ = nafi.exporter()


@export
@partial(jax.jit, static_argnames='interval_type')
def bayesian_pvals(lnl, hypotheses, interval_type='hdpi', ln_prior=None):
    """Compute Bayesian p-values from log likelihood(ratio)s

    Arguments:
     - lnl: Log likelihood(ratio)s, (trials, hypotheses)
     - hypotheses: Hypotheses to compute p-values for.
     - interval_type: Interval type for which to compute p-values, either
        'hdpi' (high-density posterior, default), or 'ul' (upper limit), 
        or 'll' (lower limit). Note 'll' will give the CDF.
     - ln_prior: Log of prior, (hypotheses) array. 
        If not provided, assumed proportional to the spacing between hypotheses,
        so the prior density is flat/uniform in the parameter of interest.
    """
    # Note this does not depend on p_n_mu... likelihood principle!

    # Estimate spacing between hypotheses
    # (gradient is ~ jnp.diff, but without changing the length)
    d_hyp = jnp.gradient(hypotheses)

    # Compute log prior
    if ln_prior is None:
        # Uniform prior over the hypotheses
        prior = d_hyp
        prior /= jnp.sum(prior)
        ln_prior = jnp.log(prior)

    # Get the posterior on our grid of mu
    # ln_evidence = logsumexp(lnl_sig, axis=-1)
    ln_posterior = lnl + ln_prior[None,:] #  - ln_evidence[...,None]
    ln_posterior -= logsumexp(ln_posterior, axis=-1)[...,None]
    posterior = jnp.exp(ln_posterior)

    if interval_type == 'hdpi':
        # Get the highest density interval
        order = jnp.argsort(posterior / d_hyp[None,:], axis=-1)
        reorder = jnp.argsort(order, axis=-1)
        # Sum posterior from low to high posterior density bins
        ps = jnp.take_along_axis(
            jnp.cumsum(
                jnp.take_along_axis(
                    posterior, order, axis=-1), 
                axis=-1),
            reorder, axis=-1)

    elif interval_type == 'ul':
        # Upper limits: p-values start at 1 and go to 0
        # sum from high to low mu
        ps = jnp.cumsum(posterior[...,::-1], axis=-1)[...,::-1]

    elif interval_type == 'll':
        # Lower limits: p-values start at 0 and go to 1
        # sum from low to high mu
        ps = jnp.cumsum(posterior, axis=-1)

    else:
        raise ValueError(f"Unknown interval type {interval_type}")

    return ps


@export
@jax.jit
def posterior_cdf(lnl, hypotheses, ln_prior=None):
    """Compute cumulative posterior density from log likelihood(ratio)s

    Arguments:
     - lnl: Log likelihood(ratio)s, (trials, hypotheses)
     - hypotheses: Hypotheses to compute posterior for.
     - ln_prior: Log of prior, (hypotheses) array. 
        If not provided, assumed proportional to the spacing between hypotheses
        so the prior is flat/uniform in the parameter of interest.
    """
    return bayesian_pvals(
        lnl, hypotheses, interval_type='ll', ln_prior=ln_prior)



@export
@jax.jit
def min_cred_ul(posterior_cdf, hypotheses, ll=0, cl=0.9):
    """Return upper limits so that the intervals with the lower limit ll 
    have credibility cl.
    
    Experimental function, may be removed later.

    Arguments:
     - posterior_cdf: posterior CDF, shape (n_outcomes, n_hypotheses,)
     - hypotheses: hypotheses, shape (n_hypotheses,)
     - ll: lower limits, shape (n_outcomes,). If omitted, uses hypotheses[0].
     - cl: credibility level, default 0.9
    """
    # Ensure ll is an array of the right shape
    n_outcomes, _ = posterior_cdf.shape
    ll = ll * jnp.ones(n_outcomes)

    inv_cred_ll = nafi.credibility(
        posterior_cdf, hypotheses, 
        hypotheses[0] * jnp.ones_like(ll), 
        ll)
    # Desired posterior quantile
    min_cred_q = (inv_cred_ll + cl).clip(0, 1)
    # UL which, together with ll, gives a credible interval
    _, ul_mincred = jax.vmap(nafi.intervals, in_axes=(0, None, None, 0))(
        # Bayesian UL p-vals = 1 - posterior CDF
        1 - posterior_cdf, 
        # hypotheses =
        hypotheses,
        # interpolate=
        True,
        # cl=
        min_cred_q)
    return ul_mincred
