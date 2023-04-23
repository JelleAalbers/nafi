"""Frequentist methods for computing p-values from likelihoods
"""
import numpy as np
from scipy import stats

import nafi

export, __all__ = nafi.exporter()

DEFAULT_TEST_STATISTIC = 't'
DEFAULT_INTERPOLATE_BESTFIT = False
DEFAULT_CLS = False


@export
def maximum_likelihood(lnl, interpolate=True):
    """Return maximum likelihood, and index of best-fit hypothesis
    
    Returns two (|trials|, |n_sig|) arrays: the bestfit likelihood ratio and
    index of best-fit hypothesis.
    """
    # Find the maximum likelihood

    best_i_coarse = np.argmax(lnl, axis=-1)
    lnl_best_coarse = np.max(lnl, axis=-1)
    if not interpolate:
        return lnl_best_coarse, best_i_coarse

    # Just doing linear interpolation won't find a better fit;
    # it would always find the same point as the coarse search.
    # All the work, if any, is done by np.gradient here.

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
def test_statistic(
        lnl, 
        statistic=DEFAULT_TEST_STATISTIC, 
        interpolate_bestfit=DEFAULT_INTERPOLATE_BESTFIT):
    """Return test statistic computed from likelihood curves

    Arguments:
        lnl: (|n_trials|, |n_hypotheses|) array of log likelihoods
        statistic: 't' or 'q' or 'q0' or 'lep'
            * t is L(hypothesis)/L(bestfit)
            * q is t, but zero if hypothesis <= bestfit (data is an excess), 
            * q0 is L(bestfit)/L(0)
            * lep is L(hypothesis)/L(0)
        interpolate_bestfit: If True, use interpolation to estimate 
            the best-fit hypothesis more precisely.

    Returns:
        Array of test statistics, same shape as lnl
    """
    _, n_hyp = lnl.shape

    lnl_best, best_i = maximum_likelihood(lnl, interpolate=interpolate_bestfit)

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
        raise ValueError(f"Unknown statistic {statistic}")
    return ts


@export
def asymptotic_pvals(ts, statistic=DEFAULT_TEST_STATISTIC, cls=DEFAULT_CLS):
    """Compute asymptotic frequentist p-value for test statistics ts
    
    Arguments:
        ts: array of test statistics
        statistic: 't' or 'q'
        cls: If True, use asymptotic formulae appropriate for the CLs method 
            instead of those for a frequentist/Neyman construction.
            (Currently just raises NotImplementedError)
    """
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
    return ps


@export
def neyman_pvals(ts, toy_weight, freeze_truth_index=None, progress=False):
    """Compute p-values from test statistics ts using a Neyman construction"""
    n_hyp = ts.shape[-1]
    ps = np.zeros(ts.shape)
    for hyp_i in nafi.utils.tqdm_maybe(progress)(
            range(n_hyp), desc='Computing p-values', leave=False):
        if freeze_truth_index is not None:
            truth_i = freeze_truth_index
        else:
            truth_i = hyp_i
        ps[...,hyp_i] = 1 - (
            nafi.utils.weighted_ps(
                ts[:,hyp_i], 
                toy_weight[:,truth_i]
            ).reshape(ts.shape[:-1]))
    return ps


@export
def cls_pvals(ts, toy_weight, progress=False):
    """Compute p-value ratios used in the CLs method. 
    First hypothesis must be background-only.
    """
    ps = neyman_pvals(ts, toy_weight, progress=progress)
    ps_0 = neyman_pvals(ts, toy_weight, freeze_truth_index=0, progress=progress)
    # PDG review has a 1- in the denominator... hm... 
    # Don't see it in Read and Junk's original papers...??
    return ps / ps_0


# Higher level API, combining the above functions
@export
def ts_and_pvals(
        lnl, 
        toy_weight, 
        statistic=DEFAULT_TEST_STATISTIC, 
        cls=DEFAULT_CLS,
        interpolate_bestfit=DEFAULT_INTERPOLATE_BESTFIT, 
        asymptotic=False,
        progress=False):
    """Compute frequentist test statistics and p-values for likelihood ratios
    
    Parameters
    ----------
    lnl : array
        Likelihood ratio, shape (|trials|, |n_hypotheses|)
    toy_weight : array
        (Hypothesis-dependent) weight of each toy, shape (|trials|, |n_hyp|)
    interpolate_bestfit : bool
        If True, use interpolation to estimate the best-fit hypothesis more
        precisely.
    asymptotic: bool
        If True, use asymptotic approximation for the test statistic 
        distribution.
    cls: False
        If True, use the CLs method instead of a Neyman construction.
        Hypothesis 0 (the first one) must be the background-only hypothesis.

    Returns: (ts, ps), arrays of test statistics and p-values 
        of the same shape as lnl.
    """
    # Compute statistics
    ts = test_statistic(lnl, statistic, interpolate_bestfit=interpolate_bestfit)

    # Convert stats to p-values
    if asymptotic:
        ps = asymptotic_pvals(ts, statistic, cls=cls)
    elif cls:
        ps = cls_pvals(ts, toy_weight, progress=progress)
    else:
        ps = neyman_pvals(ts, toy_weight, progress=progress)
    return ts, ps


@export
def single_ts_and_pvals(
        lnl_obs, 
        ts=None,
        ps=None,
        statistic=DEFAULT_TEST_STATISTIC, 
        interpolate_bestfit=DEFAULT_INTERPOLATE_BESTFIT,
        asymptotic_cls=DEFAULT_CLS,
        asymptotic=False):
    """Compute test statistic and p-value for a single outcome's likelihood

    Returns: (ts, ps), arrays of test statistics and p-values, 
        shape is (|n_hypotheses|).
    
    Arguments:
     - lnl_obs: (|n_hyps|,) array of log likelihoods for one trial
     - ts: (|n_toys|, |n_hyps|,) array of toy test statistics
     - ps: (|n_toys|, |n_hyps|,) array of toy p-values
     - asymptotic: If True, use asymptotic distribution of the test statistic
     - asymptotic_cls: If True, CLs asymptotics are used if asymptotic=True
        (Argument is not called 'cls' to avoid people feeding in Neyman p-values
         and thinking setting this gets them Cls results.)
    """
    # Compute test statistic
    # shape: (1, |n_hypotheses|)
    t_obs = test_statistic(
        lnl_obs[None,:], 
        statistic=statistic, 
        interpolate_bestfit=interpolate_bestfit)
    if asymptotic:
        p_obs = asymptotic_pvals(t_obs, statistic=statistic, cls=asymptotic_cls)
    else:
        assert ts is not None and ps is not None, "Need toy test statistics and p-values"
        # Get index of closest toy trial (for each hypothesis)
        # shape: (|n_hypotheses|)
        best_trial = np.argmin(np.abs(ts - t_obs), axis=0)
        # Get corresponding p-value from ps
        # shape: (|n_hypotheses|)
        p_obs = ps[best_trial, np.arange(len(best_trial))]
    return t_obs[:,0], p_obs
