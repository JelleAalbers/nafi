"""Methods for producing likelihood ratios for a counting experiment
with background.

End goal is an (n_sig, BATCH, hypothesis_i) tensor of likelihood ratios,
where 
 - n_sig is the number of generated signal events
 - BATCH is a batch dimension
 - hypothesis_i indexes the hypothesis (signal rate)
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


import nafi
export, __all__ = nafi.exporter()


@export
def lnl_and_weights(mu_sig, mu_bg, n_max=None):
    """Return (logl, toy weight) for a counting experiment with background.

    Both are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig: Array with signal rate hypotheses
        mu_bg: Background rate (scalar)
    """
    if n_max is None:
        # Jax doesn't like this
        n_max = mu_bg + np.max(mu_sig)
        n_max = n_max + 5 * n_max**0.5 + 5
    return _lnl_and_weights(mu_sig, mu_bg, n_max)


@partial(jax.jit, static_argnames='n_max')
def _lnl_and_weights(mu_sig, mu_bg, n_max):
    # Total expected events
    mu_tot = mu_sig + mu_bg
    # Outcomes are defined completely by the number of events, n
    n = jnp.arange(n_max + 1, dtype=jnp.int32)
    # Probability and log likelihood of observation (given hypothesis)
    # and log likelihood (given hypothesis)
    # (n, mu) array
    lnl = jax.scipy.stats.poisson.logpmf(n[:,None], mu_tot[None,:])
    p = jnp.exp(lnl)
    # Ensure ps are normalized over n
    p /= p.sum(axis=0)

    # Log likelihood is now easy. 
    # Log(0) = -inf, which will work fine, so suppress the division warning
    #with np.errstate(divide='ignore'):
    #    lnl = np.log(p)

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
