"""TODO: this still needs work. See ``twobin`` for a more complete profile
likelihood example.
"""

import jax
import jax.numpy as jnp

import nafi
export, __all__ = nafi.exporter()


@export
def conditional_bestfit_bg(n, mu_sig, mu_bg_estimate, sigma_bg):
    be = mu_bg_estimate
    sigma = sigma_bg
    mu = mu_sig
    b = 0.5 * (
        be - mu - sigma**2
        + ( (be+mu)**2 - 2 * (be - 2 * n + mu) * sigma**2 + sigma**4 )**0.5)
    b = jnp.where(jnp.isfinite(b) & (b >= 0), b, 0)
    return b


@export
def lnl_and_weights(mu_sig_hyp, mu_bg_true, mu_bg_model, sigma_bg, n_max=None):
    """Return (logl, toy_weight) for a counting experiment with a background
    that has a Gaussian absolute theoretical uncertainty on mu_bg of
    sigma_bg.

    Here `lnl` and `toy_weight` are both (n_outcomes, hypotheses) arrays,
        where lnl contains the log likelihood at each hypothesis
        with the background rate profiled out,
        and toy_weight contains P(outcome | hypotheses), normalized over
        outcomes.

    Since the background cannot be negative, this model can be ill-defined.
    Consider using ``nafi.likelihoods.onoff`` instead.

    Arguments:
        mu_sig_hyp: Array with signal rate hypotheses
        mu_bg_true: True background rate to assume for toy data,
            array of same shape as mu_sig_hyp.
        mu_bg_model: Expected/modelled mu_bg, scalar
        sigma_bg: Gaussian absolute uncertainty on mu_bg_model
            (i.e. in number of events, not a percentage)
    """
    if n_max is None:
        n_max = nafi.large_n_for_mu(mu_bg_true.max() + mu_sig_hyp.max())
    return _lnl_and_weights(mu_sig_hyp, mu_bg_true, mu_bg_model, sigma_bg, n_max)


@export
def _lnl_and_weights(mu_sig_hyp, mu_bg_true, mu_bg_estimate, sigma_bg, n_max):
    # Total expected events
    mu_tot_true = mu_sig_hyp + mu_bg_true
    # Outcomes defined by n
    n = jnp.arange(n_max)
    # Probability of observation (given hypothesis and mu_bg_true)
    # (n, mu) array
    p = jax.scipy.stats.poisson.pmf(n[:,None], mu_tot_true[None,:])
    # Ensure ps are normalized over n
    p /= p.sum(axis=0)

    # Bestfit background rate (analytic solution)
    # (n, mu) array
    b = conditional_bestfit_bg(
        n[:,None], mu_sig_hyp[None,:],
        mu_bg_estimate, sigma_bg)

    # Log likelihood
    # lnl = (
    #     jax.scipy.stats.poisson.logpmf(n[:,None], mu_sig_hyp[None,:] + b)
    #     - (b - mu_bg_estimate)**2 / (2 * sigma_bg**2))
    lnl = (
        -(mu_sig_hyp[None,:] + b)
        + jax.scipy.special.xlogy(n[:,None], mu_sig_hyp[None,:] + b)
        - (b - mu_bg_estimate)**2 / (2 * sigma_bg**2))

    return lnl, p
