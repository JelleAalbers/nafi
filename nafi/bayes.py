import numpy as np
from scipy.special import logsumexp

import nafi

export, __all__ = nafi.exporter()


@export
def bayesian_pvals(lnl, interval_type='hdpi', prior=None, hypothesis_spacing=None):
    """Compute Bayesian p-values for likelihood ratios
    
    These will be the cumulative sum of the posterior in reverse order of the
    interval construction scheme (e.g. low-density gets low p under HDPI).
    """
    # Note this does not depend on p_n_mu... likelihood principle!

    n_hyp = lnl.shape[-1]

    # Convert likelihoods to posterior
    n_hyp = lnl.shape[-1]
    if prior is None:
        # Uniform prior over the hypotheses
        prior = np.ones(n_hyp) / n_hyp
    ln_prior = np.log(prior)
    # Get the posterior on our grid of mu
    # ln_evidence = logsumexp(lnl_sig, axis=-1)
    ln_posterior = lnl + ln_prior[None,:] #  - ln_evidence[...,None]
    posterior = np.exp(ln_posterior - logsumexp(ln_posterior, axis=-1)[...,None])

    if interval_type == 'hdpi':
        # Get the highest density interval
        if hypothesis_spacing is None:
            # Assume hypotheses we calculated on were equidistant.
            # (or maybe they were not, but the user didn't give a prior, so we made
            #  a flat prior and it all cancels out... maybe??)
            order = np.argsort(posterior, axis=-1)
        else:
            order = np.argsort(posterior / hypothesis_spacing[None,:], axis=-1)
        reorder = np.argsort(order, axis=-1)
        # Sum posterior from low to high posterior density bins
        ps = np.take_along_axis(
            np.cumsum(
                np.take_along_axis(
                    posterior, order, axis=-1), 
                axis=-1),
            reorder, axis=-1)
        
    elif interval_type == 'ul':
        # Upper limits: sum from high to low values
        ps = np.cumsum(posterior[...,::-1], axis=-1)[...,::-1]

    else:
        raise ValueError(f"Unknown interval type {interval_type}")

    return ps
