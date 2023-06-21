"""The on-off / two-bin counting experiment.

n events are observed in the main experiment, m in the ancilla.

The main experiment has signal mu_sig and background mu_bg.

The ancilla only has background mu_bg * tau
"""
import jax
from jax import numpy as jnp


# Can't jit this guy, it produces arrays whose shape depend on the args
def outcomes(n_max, m_max, ravel=True):
    """Return 2-tuple of two ((n_max + 1) * (m_max + 1)) arrays with possible
    experimental outcomes.

    Arguments:
        - n_max: Maximum number of events in main experiment to consider
        - m_max: Maximum number of events in ancillary experiment to consider
        - ravel: If False (default is true), instead return two 
            (n_max + 1, m_max + 1) 2d arrays.
    """
    # Outcomes defined by n (obs in main) and m (obs in anc)
    n = jnp.arange(n_max + 1)
    m = jnp.arange(m_max + 1)
    # (n, m) grids with n and m values
    n, m = jnp.meshgrid(n, m, indexing='ij')
    if ravel:
        # Ravel into flat (n * m) arrays
        n, m = n.ravel(), m.ravel()
    return n, m


@jax.jit
def profile_lnl(mu_sig_hyp, n, m, *, tau):
    """Return (n_outcomes, n_hyp) profile log likelihood
    """
    # Get best-fit background (n_outcomes, n_hyp)
    b_doublehat = conditional_bestfit_bg(
        mu_sig_hyp[None,:], n[:,None], m[:,None], tau=tau)
    
    return _lnl(mu_sig_hyp[None,:], b_doublehat, n[:,None], m[:,None], tau=tau)


@jax.jit
def _lnl(mu_sig_hyp, mu_bg, n, m, *, tau):
    """Return (n_outcomes, n_hyp) log likelihood,
    without unnecessary factorial term.

    All arguments must be broadcastable
    """
    mu_main = mu_sig_hyp + mu_bg
    mu_anc = tau * mu_bg
    return (
        -mu_main + jax.scipy.special.xlogy(n, mu_main)
        -mu_anc + jax.scipy.special.xlogy(m, mu_anc)
    )


@jax.jit
def profile_weights(mu_sig_hyp, n, m, n_obs, m_obs, *, tau):
    # Get best-fit background, (hypotheses,) array
    b_doublehat = conditional_bestfit_bg(
        mu_sig_hyp, n_obs, m_obs, tau=tau)

    # Get P(outcome | hypothesis); (outcome, hypothesis) array
    p_outcome = _p_outcome(
        mu_sig_hyp[None,:],
        b_doublehat[None,:],
        n[:,None],
        m[:,None],
        tau=tau)
    # TODO: investigate where these are coming from
    p_outcome = jnp.nan_to_num(p_outcome)
    # Ensure weights of outcomes sum to one
    p_outcome /= p_outcome.sum(axis=0)
    return p_outcome


@jax.jit
def true_weights(mu_sig_hyp, mu_bg_hyp, n, m, *, tau):
    """Return (outcomes, signal hypotheses, background hypotheses) array 
    with weights of outcomes given the true signal and background hypotheses.

    (i.e. P(outcome | sig, bg), normalized to sum to 1 over outcomes)
    
    Arguments:
     - mu_sig_hyp: Array with signal rate hypotheses
     - mu_bg_hyp: Array with background rate hypotheses
     - n, m, tau: same as always
    """
    p_outcome = _p_outcome(
        mu_sig_hyp[None,:,None], 
        mu_bg_hyp[None,None,:], 
        n[:,None,None], 
        m[:,None,None],
        tau=tau)
    # Ensure outcome weights sum to one
    p_outcome /= p_outcome.sum(axis=0)
    return p_outcome


@jax.jit
def _p_outcome(mu_sig, mu_bg, n, m, *, tau):
    return (
        jax.scipy.stats.poisson.pmf(n, mu_sig + mu_bg)
        * jax.scipy.stats.poisson.pmf(m, tau * mu_bg))


@jax.jit
def conditional_bestfit_bg(mu_sig_hyp, n, m, *, tau=1):
    """Return the conditional best fit background rate for an on-off experiment.
    See lnl_and_weights for details.

    with unknown background.

    Arguments:
        n: Number of observed events
        m: Number of observed events in the ancillary experiment
        mu_hyp: Array with signal rate hypotheses
        tau: Ratio of background rates in the two experiments

    See e.g. Rolke, Lopez, Conrad (2005) arxiv:0403059,
    or Sen, Walker and Woodrofe (2009)
    """
    _q = n + m - (1+tau) * mu_sig_hyp
    root1 = (
        (_q + (_q**2 + 4 * (1 + tau) * m * mu_sig_hyp)**0.5)
        /(2*(1+tau)))
    # Probably this root is never taken. Oh well.
    # root2 = (
    #     (_q - (_q**2 + 4 * (1 + tau) * m * mu_hyp)**0.5)
    #     /(2*(1+tau)))
    return root1
