import numpy as np
from scipy import stats
from scipy.special import xlogy

import nafi
export, __all__ = nafi.exporter()


@export
def conditional_bestfit_bg(n, m, mu_hyp, tau=1):
    """Return the conditional best fit background rate for a counting experiment
    with unknown background.

    Arguments:
        n: Number of observed events
        m: Number of observed events in the ancillary experiment
        mu_hyp: Array with signal rate hypotheses
        tau: Ratio of background rates in the two experiments

    See e.g. Rolke, Lopez, Conrad (2005), arxiv:0403059
    """
    _q = n + m - (1+tau) * mu_hyp
    root1 = (
        (_q + (_q**2 + 4 * (1 + tau) * m * mu_hyp)**0.5)
        /(2*(1+tau)))
    # Probably this root is never taken. Oh well.
    # root2 = (
    #     (_q - (_q**2 + 4 * (1 + tau) * m * mu_hyp)**0.5)
    #     /(2*(1+tau)))
    return root1


@export
def get_lnl(mu_sig_hyp, mu_bg_true, tau=1):
    """Return (logl, toy weight) for a counting experiment with 
        unknown background. The background is constrained by another 
        counting experiment, which measures tau times the same background.

    Both results are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig_hyp: Array with signal rate hypotheses
        mu_bg: True background for toys, array of same shape as mu_sig_hyp
        tau: Ratio of background rates in the two experiments
    """
    # Total expected events

    # Outcomes defined by n (obs in main) and m (obs in anc)
    # + 2 just in case some clown tries tau = 0 etc.
    n = np.arange(stats.poisson(mu_sig_hyp.max() + mu_bg_true.max()).ppf(0.999) + 2)
    m = np.arange(stats.poisson(tau * mu_bg_true.max()).ppf(0.999) + 2)
    # (n, m)
    n, m = np.meshgrid(n, m, indexing='ij')
    # (n, m, n_hyp)
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
    # (imagine running an MC with many toys,
    #  setting true nuisances to mu_bg_true)
    p_outcome = (
        stats.poisson(mu_sig_hyp[None,None,:] + mu_bg_true[None,None,:]).pmf(n)
        * stats.poisson(tau * mu_bg_true[None,None,:]).pmf(m))
    
    # Log likelihood is now easy...
    # NO!!! mu_bg is different!!
    # lnl = np.log(p_outcome)

    # Flatten outcomes
    lnl = lnl.reshape(-1, len(mu_sig_hyp))
    p_outcome = p_outcome.reshape(-1, len(mu_sig_hyp))

    # Ensure outcome weights sum to one
    p_outcome /= p_outcome.sum(axis=0)

    return lnl, p_outcome
