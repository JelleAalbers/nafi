"""Methods for producing likelihood ratios for a counting experiment
with background.

End goal is an (n_sig, BATCH, hypothesis_i) tensor of likelihood ratios,
where 
 - n_sig is the number of generated signal events
 - BATCH is a batch dimension
 - hypothesis_i indexes the hypothesis (signal rate)
"""

import numpy as np
from scipy import stats

import nafi
export, __all__ = nafi.exporter()


def get_llr(mu, mu_bg):
    # Total expected events
    mu_tot = mu + mu_bg
    # best-fit total mu
    n = np.arange(stats.poisson(mu.max()).ppf(0.999)).astype(int)
    mu_tot_best = (n - mu_bg).clip(0, None) + mu_bg
    # Make (mu, n) arrays
    o = None
    # Likelihood ratio of observation (given hypothesis)
    t = ( 
        # Log P(test)
        stats.poisson(mu_tot[:,o]).logpmf(n[o,:]) 
        # Log P(best fit)
        - stats.poisson(mu_tot_best[o,:]).logpmf(n[o,:]) )
    # Probability of observation (given hypothesis)
    p = stats.poisson(mu_tot[:,o]).pmf(n[o,:])
    # Insert batch dimension of 1: no need to MC event positions since
    # we're just counting!
    return t[o,:,:], p
