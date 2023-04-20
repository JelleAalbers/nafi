"""Methods for converting likelihood ratios to p-values
(which are then used for limit setting)
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp

import nafi

export, __all__ = nafi.exporter()


@export
def bayesian_p(lnl, interval_type='hdpi', prior=None, hypothesis_spacing=None):
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
    np.testing.assert_allclose(np.sum(posterior, axis=-1), 1)

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



##
# Frequentist inference
##

@export
def bestfit(lnl, interpolate=True):
    """Return best-fit likelihood ratio and index of best-fit hypothesis
    
    Returns two (|trials|, |n_sig|) arrays: the bestfit likelihood ratio and
    index of best-fit hypothesis.
    """
    # Find the maximum likelihood

    best_i_coarse = np.argmax(lnl, axis=-1)
    lnl_best_coarse = np.max(lnl, axis=-1)
    if not interpolate:
        return lnl_best_coarse, best_i_coarse

    # Maybe this is slightly faster than the line above. 
    # Not sure / not sure it matters.
    # lnl_best_coarse = np.take_along_axis(lnl_sig, best_i_coarse, axis=-1)            

    # Estimate the maximum likelihood slightly more refined using interpolation
    # Probably this makes absolutely zero difference, 
    # since the likelihood hardly changes near its minimum
    # TODO: use autodiff instead of finite differences
    d_lnl_dmuindex = np.gradient(lnl, axis=-1)
    best_i_fine = nafi.utils.find_root_vec(y=d_lnl_dmuindex, guess_i=best_i_coarse)
    grad_at_coarse_best = np.take_along_axis(
        arr=d_lnl_dmuindex, 
        indices=best_i_coarse[...,None], 
        axis=-1)[...,0]
    lnl_best = lnl_best_coarse + (best_i_fine % 1) * np.abs(grad_at_coarse_best)
    return lnl_best, best_i_fine


@export
def neyman_pvals(ts, toy_weight, on_bgonly=False, progress=False):
    """Convert ts to p-values using a Neyman construction"""
    n_hyp = ts.shape[-1]
    ps = np.zeros(ts.shape)
    for hyp_i in nafi.utils.tqdm_maybe(progress)(
            range(n_hyp), desc='Computing p-values', leave=False):
        if on_bgonly:
            # Assuming hypothesis 0 is background only!
            truth_i = 0
        else:
            truth_i = hyp_i
        ps[...,hyp_i] = 1 - (
            nafi.utils.weighted_ps(
                ts[:,hyp_i], 
                toy_weight[:,truth_i]
            ).reshape(ts.shape[:-1]))
    return ps


@export
def frequentist_p(
        lnl, 
        toy_weight, 
        statistic='t', 
        cls=False,
        interpolate_bestfit=True, 
        asymptotic=False,
        progress=False):
    """Compute frequentist p-values for likelihood ratios
    
    Parameters
    ----------
    lnl : array
        Likelihood ratio, shape (|trials|, |n_hyp|)
    toy_weight : array
        (Hypothesis-dependent) weight of each toy, shape (|trials|, |n_hyp|)
    interpolate_bestfit : bool
        If True, use interpolation to estimate the best-fit hypothesis more
        precisely.
    asymptotic: bool
        If True, use asymptotic approximation for the test statistic 
        distribution.
    cls: False
        If True, use the dreaded CLs method.
    """
    _, n_hyp = lnl.shape

    lnl_best, best_i = bestfit(lnl, interpolate=interpolate_bestfit)

    # Compute statistics (|n_trials|,|n_hyps|)
    if statistic in ('t', 'q'):
        ts = -2 * (lnl - lnl_best[...,None])
        if statistic == 'q':
            # Zero excesses, i.e. hypothesis <= bestfit
            ts *= best_i[...,None] <= np.arange(n_hyp)[None,:]
    elif statistic == 'q0':
        # L(best)/L(0). Note this does not depend on the hypothesis.
        # Assuming hypothesis 0 is background only!
        ts = -2 * (lnl_best[...,None] - lnl[...,0][...,None])
        # This is a sad memory hog, but the rest of the code is kind of stupid
        # so we need it to work
        ts = ts.repeat(n_hyp, axis=-1)
    elif statistic == 'lep':
        # L(test)/L(0)
        ts = -2 * (lnl - lnl[...,0][...,None])
    else:
        raise ValueError(f"Unknown stat {statistic}")

    ##
    # Convert stats to p-values
    ##

    # P(t(mu)|mu), (|n_sig|,|trials|,|mu|)
    # ps = np.zeros(ts.shape)     # P under signal + background

    if asymptotic:
        # Compute asymptotic p-values
        if statistic not in ('t', 'q'):
            raise ValueError("Don't know the asymptotic distribution "
                             f"of statistic {statistic}")
        if cls:
            raise NotImplementedError(
                "CLs asymptotics... might as well join LHC at this point")
        # q's distribution has a delta function at 0, but since 0 is the lowest
        # possible value, it never contributes to the survival function.
        # The special function scipy uses here is quite expensive.
        ps = stats.chi2(df=1).sf(ts)
        if statistic == 'q':
            ps *= 0.5
    
    else:
        # Do neyman construction
        ps = neyman_pvals(ts, toy_weight, 
                          progress=progress)
            
        if cls:
            ps_0 = neyman_pvals(ts, toy_weight, on_bgonly=True, 
                                progress=progress)
            # PDG review has a 1- in the denominator... hm... 
            # Don't see it in Read and Junk...??
            ps = ps / ps_0

    return ps
