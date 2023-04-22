import numpy as np
from scipy import stats
from scipy.special import xlogy

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
    b = np.where(np.isfinite(b) & (b >= 0), b, 0)
    return b


@export
def get_lnl(mu_sig_hyp, mu_bg_true, mu_bg_estimate, sigma_bg):
    """Return (logl, toy weight) for a counting experiment with a background
        that has a Gaussian absolute theoretical unceratainty sigma_bg.

    Both are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig_hyp: Array with signal rate hypotheses
        mu_bg_true: True background rate, array of same shape as mu_sig_hyp 
        mu_bg_estimate: Estimated background rate, scalar
        sigma_bg: Gaussian absolute uncertainty on background
    """
    # Total expected events
    mu_tot = mu_sig_hyp + mu_bg_true
    # Outcomes defined by n
    n = np.arange(stats.poisson(mu_tot.max()).ppf(0.999)).astype(int)
    # Probability of observation (given hypothesis)
    # (n, mu) array
    p = stats.poisson(mu_tot[None,:]).pmf(n[:,None])

    # Ensure ps are normalized over n
    p /= p.sum(axis=0)

    # Bestfit background rate (analytic solution)
    # (n, mu) array
    b = conditional_bestfit_bg(
        n[:,None], mu_sig_hyp[None,:],
        mu_bg_estimate, sigma_bg)

    # Log likelihood
    lnl = (
        -(mu_sig_hyp[None,:] + b) 
        + xlogy(n[:,None], mu_sig_hyp[None,:] + b) 
        - (b - mu_bg_estimate)**2 / (2 * sigma_bg**2))

    return lnl, p, b
