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


@export
def get_lnl(mu_sig_hyp, mu_bg):
    """Return (logl, toy weight) for a counting experiment with background.

    Both are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_sig_hyp: Array with signal rate hypotheses
        mu_bg: Background rate (scalar)
    """
    # Total expected events
    mu_tot = mu_sig_hyp + mu_bg
    # Outcomes defined by n
    n = np.arange(stats.poisson(mu_tot.max()).ppf(0.999)).astype(int)
    # Probability of observation (given hypothesis)
    # (n, mu) array
    p = stats.poisson(mu_tot[None,:]).pmf(n[:,None])
    # Log likelihood is now easy...
    lnl = np.log(p)
    # Ensure ps are normalized over n
    p /= p.sum(axis=0)
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
    lnl, _ = get_lnl(mu_sig, mu_bg)
    return lnl[n]
