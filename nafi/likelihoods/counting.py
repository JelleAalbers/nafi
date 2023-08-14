"""Methods for producing likelihood ratios for a counting experiment
with background.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


import nafi
export, __all__ = nafi.exporter()


@export
def lnl_and_weights(mu_sig, mu_bg, n_max=None, return_outcomes=False):
    """Return (logl, toy weight) for a counting experiment with background.

    Both are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig: Array with signal rate hypotheses
        mu_bg: Background rate (scalar)
        n_max: Largest number of events to consider. If None, will be
            determined automatically from mu_sig and mu_bg.
        return_outcomes: If True, return a third array of shape (n_outcomes,)
            containing the number of events for each outcome.
    """
    if n_max is None:
        # Can't do dynamic shapes inside jitted jax functions, so we have to
        # fix a max n in this ugly wrapper function
        n_max = nafi.large_n_for_mu(mu_bg + np.max(mu_sig))
    return _lnl_and_weights(mu_sig, mu_bg, n_max, return_outcomes)


@partial(jax.jit, static_argnames=('n_max', 'return_outcomes'))
def _lnl_and_weights(mu_sig, mu_bg, n_max, return_outcomes=False):
    # Total expected events
    mu_tot = mu_sig + mu_bg
    # Outcomes are defined completely by the number of events, n
    n = jnp.arange(n_max + 1, dtype=jnp.int32)
    # Probability and log likelihood of observation (given hypothesis)
    # (n, mu) arrays
    lnl = jax.scipy.stats.poisson.logpmf(n[:,None], mu_tot[None,:])
    p = nafi.lnl_to_weights(lnl)
    if return_outcomes:
        return lnl, p, n
    return lnl, p


@export
def single_lnl(*, n, mu_sig, mu_bg):
    """Return log likelihood for a single counting experiment observation

    Arguments:
        n: Observed number of events
        mu_sig: Signal rate hypothesis (array)
        mu_bg: Background rate (scalar)
    """
    # This is slow... but it doesn't matter for this simple toy.
    lnl, _ = lnl_and_weights(mu_sig, mu_bg)
    return lnl[n]
