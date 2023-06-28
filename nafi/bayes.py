from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import nafi

export, __all__ = nafi.exporter()


@export
@jax.jit
def hypothesis_spacing(hypotheses):
    d_hyp = jnp.gradient(hypotheses)
    # Extreme hypotheses represent only half as much parameter space
    # as the other: not [x-dx/2, x+dx/2] but [x, x + dx/2] or [x - dx/2, x]
    # TODO: should take account of nonuniform gradient, so can't just do /2
    d_hyp = d_hyp.at[0].set(d_hyp[0] / 2)
    d_hyp = d_hyp.at[-1].set(d_hyp[-1] / 2)
    return d_hyp


@export
@jax.jit
def uniform_prior(hypotheses):
    """Return prior over hypotheses corresponding to a uniform prior density 
    over the parameter of interest.

    Specifically, the return value represents the following probabilities:

    .. math::
        P(h - dh/2 <= truth <= h + dh/2)

    for interior hypotheses;

    .. math::
        P(h <= truth <= h + dh/2)

    for the first hypothesis;

    .. math::
        P(h - dh/2 <= truth <= h])

    for the last hypothesis. Here dh is the gradient/spacing between hypotheses.
    """
    prior = hypothesis_spacing(hypotheses)
    prior /= jnp.sum(prior)
    return prior


@export
@jax.jit
def posterior(lnl, hypotheses, ln_prior=None):
    """Compute posterior from log likelihood(ratio)s

    Arguments:
      lnl: Log likelihood(ratio)s, (trials, hypotheses)
      hypotheses: Hypotheses to compute p-values for.
      ln_prior: Log of prior, (hypotheses) array. 
        If not provided, assumed proportional to the spacing between hypotheses,
        so the prior density is flat/uniform in the parameter of interest.

    Like the prior, this accounts for spacing between hypotheses.
    """
    # Estimate spacing between hypotheses
    # (gradient is ~ jnp.diff, but without changing the length)
    
    if ln_prior is None:
        ln_prior = jnp.log(uniform_prior(hypotheses))

    # Get the posterior on our grid of mu
    # ln_evidence = logsumexp(lnl_sig, axis=-1)
    ln_posterior = lnl + ln_prior[None,:] #  - ln_evidence[...,None]
    ln_posterior -= logsumexp(ln_posterior, axis=-1)[...,None]
    posterior = jnp.exp(ln_posterior)
    return posterior


@export
@jax.jit
def posterior_cdf(posterior):
    """Compute cumulative posterior, assuming the hypotheses are a discrete
    approximation for a continuous parameter.

    In particular, we assume
     
    .. math::
       P(truth \leq h) = P(truth < h)

    and the returned posterior CDF represents:

    .. math::
        P(h - dh/2 < truth < h + dh/2)

    for interior hypotheses;

    .. math::
        P(h < truth < h + dh/2)

    for the first hypothesis, and

    .. math::
        P(h - dh/2 < truth < h])

    for the last hypothesis. Here dh is the gradient/spacing between hypotheses.
    """
    # This computes P(truth <= max[x + dx/2, hyp[-1]])
    posterior_cdf = jnp.cumsum(posterior, axis=-1)
    # For the final h this is already the result.
    # For interior h, we must subtract half the posterior at x.
    posterior_cdf = posterior_cdf.at[...,:-1].add(-0.5 * posterior[...,:-1])
    # For the first h, we must subtract P(truth <= x + dx/2),
    #                  i.e. the _full_ posterior, so P(truth <= h[0]) = 0.
    posterior_cdf = posterior_cdf.at[...,0].set(0)
    return posterior_cdf


@export
@partial(jax.jit, static_argnames='interval_type')
def bayesian_pvals(lnl, hypotheses, interval_type='hdpi', ln_prior=None):
    """Compute Bayesian p-values from log likelihood(ratio)s, suitable
    for use in nafi.intervals.

    Arguments:
      lnl: Log likelihood(ratio)s, (trials, hypotheses)
      hypotheses: Hypotheses to compute p-values for.
      interval_type: Interval type for which to compute p-values, either
        'hdpi' (high-density posterior, default), or 'ul' (upper limit), 
        or 'll' (lower limit). Note 'll' will give the CDF.
      ln_prior: Log of prior, (hypotheses) array. 
        If not provided, assumed proportional to the spacing between hypotheses,
        so the prior density is flat/uniform in the parameter of interest.
    """
    # Note this function is independent of toy weights... likelihood principle!

    post = posterior(lnl, hypotheses, ln_prior=ln_prior)

    if interval_type == 'hdpi':
        # Get the highest density interval
        # TODO: needs testing for off-by-half errors, likely has one

        # Order hypotheses from low to high posterior density
        d_hyp = hypothesis_spacing(hypotheses)
        order = jnp.argsort(post / d_hyp[None,:], axis=-1)
        reorder = jnp.argsort(order, axis=-1)

        # Posterior on hypotheses in density order (staring with least likely)
        post_ordered = jnp.take_along_axis(post, order, axis=-1)
        # ps, with hypotheses in density order
        # TODO: think carefully about handling of extreme hyps someday
        ps_ordered = nafi.posterior_cdf(post_ordered)
        # ps, with hypotheses in original order
        ps = jnp.take_along_axis(ps_ordered, reorder, axis=-1)
        #ps += 0.5 * post
        # Correct for using cumsum to approximate the CDF
        # (as in posterior_cdf)
        #ps = _posterior_cdf_cumsum_correction(ps, post)

    elif interval_type == 'ul':
        ps = 1 - posterior_cdf(post)

    elif interval_type == 'll':
        ps = posterior_cdf(post)

    else:
        raise ValueError(f"Unknown interval type {interval_type}")

    return ps


@export
@jax.jit
def posterior_cdf_quantile(posterior_cdf, hypotheses, quantile):
    """Return hypotheses where posterior CDF reaches a quantile
    
    Arguments:
      posterior_cdf: (outcomes, hypotheses) array
      hypotheses: hypotheses, shape (n_hypotheses,)
      quantile: quantile to find, (outcomes,) array
        0 <= quantile <= 1
    """
    return jax.vmap(nafi.intervals, in_axes=(0, None, None, 0))(
            1 - posterior_cdf,
            # hypotheses =
            hypotheses,
            # interpolate=
            True,
            # cl=
            quantile)[1]

@export
@jax.jit
def min_cred_ul(posterior_cdf, hypotheses, ll, cl=0.9):
    """Return upper limits so that the intervals with the lower limit ll 
    have credibility cl.
    
    Experimental function, may be removed later.

    Arguments:
      posterior_cdf: posterior CDF, shape (n_outcomes, n_hypotheses,)
        See nafi.posterior_cdf, don't just cumsum your posterior if you care 
        about off-by-half errors.
      hypotheses: hypotheses, shape (n_hypotheses,)
      ll: lower limits, shape (n_outcomes,).
      cl: credibility level, default 0.9
    """
    # Credibility of (0, ll)
    inv_cred_ll = nafi.credibility(
        posterior_cdf,
        hypotheses, 
        hypotheses[0] + 0 * ll,
        ll)
        
    # Desired posterior quantile for the ul
    min_cred_q = (inv_cred_ll + cl).clip(0, 1)

    return posterior_cdf_quantile(posterior_cdf, hypotheses, min_cred_q)


@export
@jax.jit
def repair_cred(posterior_cdf, hypotheses, ul, ll, cl=0.9):
    """Widen (ul, ll) to have credibility at least cl; 
    first by lowering the ll then raising then ul if needed.

    (That way the overcoverage is concentrated on lower hypotheses)
    
    Experimental function, may be removed later.

    Arguments:
      posterior_cdf: posterior CDF, shape (n_outcomes, n_hypotheses,)
        See nafi.posterior_cdf, don't just cumsum your posterior if you care 
        about off-by-half errors.
      hypotheses: hypotheses, shape (n_hypotheses,)
      ul: upper limits, shape (n_outcomes,).
      ll: lower limits, shape (n_outcomes,).
      cl: credibility level, default 0.9
    """
    # Credibility of (0, ul)
    cred_ul = nafi.credibility(posterior_cdf, hypotheses, ll * 0 + hypotheses[0], ul)

    # Posterior CDF quantile where we should place the ll
    desired_q_ll = (cred_ul - cl).clip(0, 1)
    ll = jnp.minimum(
        ll,
        posterior_cdf_quantile(posterior_cdf, hypotheses, desired_q_ll)
    )

    # Credibility of (0, ll)
    inv_cred_ll = nafi.credibility(
        posterior_cdf,
        hypotheses, 
        hypotheses[0] + 0 * ll,
        ll)
    
    desired_q_ul = (inv_cred_ll + cl).clip(0, 1)
    ul = jnp.maximum(
        ul,
        posterior_cdf_quantile(posterior_cdf, hypotheses, desired_q_ul)
    )
    return ll, ul
