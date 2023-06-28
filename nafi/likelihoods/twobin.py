"""A two-bin counting experiment without uncertainties.

The first bin expects f_sig_1 * mu_sig signal and mu_bg_1 background.
The second bin excepts (1 - fsig_1) * mu_sig signal and mu_bg_2 background.
"""
from functools import partial

import jax
from jax import numpy as jnp

import nafi


def lnl_and_weights(mu_sig, f_sig_1, mu_bg_1, mu_bg_2, 
                    n_max=None, m_max=None, return_outcomes=False):
    """Return (logl, toy weight) for a counting experiment with background.

    Both are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig: Array with signal rate hypotheses
        f_sig_1: Fraction of signal events in first bin
        mu_bg_1: Expected background in first bin
        mu_bg_2: Expected background in second bin
        n_max, m_max: Largest number of events in bin 1 and two to consider. 
            If None, will be determined automatically from parameters.
        return_outcomes: If True, return a third argument, tuple of (n, m)
            arrays with the number of observed events for each possible outcome.
    """
    mu_1, mu_2 = total_mus(mu_sig, f_sig_1, mu_bg_1, mu_bg_2)
    if n_max is None:
        n_max = nafi.large_n_for_mu(mu_1.max())
    if m_max is None:
        n_max = nafi.large_n_for_mu(mu_2.max())
    return _lnl_and_weights(
        mu_sig, f_sig_1, mu_bg_1, mu_bg_2, 
        n_max, m_max, return_outcomes)


@partial(jax.jit, static_argnames=('n_max', 'm_max', 'return_outcomes'))
def _lnl_and_weights(
        mu_sig, f_sig_1, mu_bg_1, mu_bg_2,
        n_max, m_max, return_outcomes=False):
    # (n_outcome,) arrays with all possible outcomes
    n, m = outcomes(n_max, m_max)

    # Log likelihood, (n_outcome, n_hyp) array
    kwargs = dict(
        mu_sig=mu_sig[None,:], n=n[:,None], m=m[:,None],
        f_sig_1=f_sig_1, mu_bg_1=mu_bg_1, mu_bg_2=mu_bg_2)
    lnl = _lnl(**kwargs)

    # Weights of outcomes, also (n_outcome, n_hyp)
    weights = _weights(**kwargs)
    if return_outcomes:
        return lnl, weights, (n, m)
    return lnl, weights


# Can't jit this guy, it produces arrays whose shape depends on the args
def outcomes(n_max, m_max, ravel=True):
    """Return 2-tuple of two ((n_max + 1) * (m_max + 1)) arrays with possible
    experimental outcomes of a two-bin experiment.

    Arguments:
      n_max: Maximum number of events in first bin to consider
      m_max: Maximum number of events in second bin to consider
      ravel: If False (default is true), instead return two 
            (n_max + 1, m_max + 1) 2d arrays.
    """
    # Outcomes defined by n (obs in main) and m (obs in anc)
    n = jnp.arange(n_max + 1)
    m = jnp.arange(m_max + 1)
    # (n, m) grids with n and m values
    n, m = jnp.meshgrid(n, m, indexing='ij')
    if ravel:
        # Ravel both n and m into flat (n * m,) = (n_outcomes,) arrays
        n, m = n.ravel(), m.ravel()
    return n, m


@jax.jit
def _lnl(mu_sig, n, m, *, f_sig_1, mu_bg_1, mu_bg_2):
    """Return (n_outcomes, n_hyp) log likelihood,
    without unnecessary factorial term.

    All arguments must be broadcastable with each other
    """
    mu_1, mu_2 = total_mus(mu_sig, f_sig_1, mu_bg_1, mu_bg_2)
    return (
        -mu_1 + jax.scipy.special.xlogy(n, mu_1)
        -mu_2 + jax.scipy.special.xlogy(m, mu_2)
    )


@jax.jit
def _weights(mu_sig, n, m, *, f_sig_1, mu_bg_1, mu_bg_2):
    """Return (n_outcomes, n_hyp) weights of outcomes.

    All arguments must be broadcastable.
    """
    mu_1, mu_2 = total_mus(mu_sig, f_sig_1, mu_bg_1, mu_bg_2)
    p_outcome = (
        jax.scipy.stats.poisson.pmf(n, mu_1)
        * jax.scipy.stats.poisson.pmf(m, mu_2))
    # TODO: investigate NaNs. From mu=0, n!=0?
    p_outcome = jnp.nan_to_num(p_outcome)
    # Ensure outcome weights of outcomes sum to one
    return p_outcome / p_outcome.sum(axis=0)


@jax.jit
def total_mus(mu_sig, f_sig_1, mu_bg_1, mu_bg_2):
    """Return (mu_1, mu_2) total expected number of events in both bins"""
    mu_1 = f_sig_1 * mu_sig + mu_bg_1
    mu_2 = (1 - f_sig_1) * mu_sig + mu_bg_2
    return mu_1, mu_2
