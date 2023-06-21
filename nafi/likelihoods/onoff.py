from functools import partial
import jax
from jax import numpy as jnp
import numpy as np

xlogy = jax.scipy.special.xlogy
poisson_pmf = jax.scipy.stats.poisson.pmf

import nafi


def lnl_and_weights(
        mu_sig, mu_bg, tau=1, 
        n_max=None, m_max=None,
          return_outcomes=False):
    """Return (logl, toy weight) for a counting experiment with 
        unknown background. The background is constrained by another 
        counting experiment, which measures tau times the same background.

    Both results are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig_hyp: Array with signal rate hypotheses
        mu_bg: True background for toys, array of same shape as mu_sig_hyp
            (will not be constant when using the profile construction.)
        tau: Ratio of background rates in the two experiments.
            If tau > 1, the ancillary experiment is more constraining
            for the background than the signal.
        return_outcomes: if true, return a third array
            (2, n_outcomes) with the (n, m) values of each outcome.
    """
    if n_max is None:
        n_max = nafi.large_n_for_mu(np.max(mu_sig + mu_bg))
    if m_max is None:
        m_max = nafi.large_n_for_mu(tau * np.max(mu_bg))
    return _lnl_and_weights(mu_sig, mu_bg, tau, n_max, m_max, return_outcomes)


@partial(jax.jit, static_argnames=('n_max', 'm_max', 'return_outcomes'))
def _lnl_and_weights(mu_sig_hyp, mu_bg_true, tau, n_max, m_max, return_outcomes):

    # Total expected events

    # Outcomes defined by n (obs in main) and m (obs in anc)
    n = jnp.arange(n_max + 1)
    m = jnp.arange(m_max + 1)
    # (n, m) grids with n and m values
    n, m = jnp.meshgrid(n, m, indexing='ij')
    # reshape to (n, m, 1), third axis will represent hypotheses
    n, m = n[...,None], m[...,None]

    # Conditional best-fit background rate
    # (n, m, n_hyp)
    b_doublehat = conditional_bestfit_bg(
        n, m, mu_sig_hyp[None,None,:], tau=tau)
    
    # Total expected events
    # (n, m, n_hyp)
    mu = mu_sig_hyp[None,None,:] + b_doublehat
    mu_anc = tau * b_doublehat
    
    # log likelihood (omitting unnecessary factorial term)
    lnl = (
        -mu + xlogy(n, mu)
        -mu_anc + xlogy(m, mu_anc)
    )

    # p(outcome | hypothesis, true mu_bg)
    # (equivalent to running an MC with many toys,
    #  setting true nuisances to mu_bg_true)
    p_outcome = (
        poisson_pmf(n, mu_sig_hyp[None,None,:] + mu_bg_true[None,None,:])
        * poisson_pmf(m, tau * mu_bg_true[None,None,:]))

    # Note lnl != np.log(p_outcome), since mu_bg is different in the two cases
    # (p_outcome uses mu_bg_true, lnl uses b_doublehat)

    # Flatten outcome axes
    lnl = lnl.reshape(-1, len(mu_sig_hyp))
    p_outcome = p_outcome.reshape(-1, len(mu_sig_hyp))

    # Ensure outcome weights sum to one
    p_outcome /= p_outcome.sum(axis=0)

    if not return_outcomes:
        return lnl, p_outcome
    
    # Also return n and m, stacked in a (2, n_outcomes) array
    # Note the [...,0] to remove the shape-1 third dimension we added above
    # (for easy broadcasting over the hypothesis dimension)
    return lnl, p_outcome, jnp.stack([n[...,0].ravel(), m[...,0].ravel()])


def conditional_bestfit_bg(n, m, mu_sig_hyp, tau=1):
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
