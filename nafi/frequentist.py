"""Frequentist methods for computing p-values from likelihoods
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import nafi

export, __all__ = nafi.exporter()

DEFAULT_TEST_STATISTIC = 't'
DEFAULT_INTERPOLATE_BESTFIT = True
DEFAULT_CLS = False


@export
@partial(jax.jit, static_argnames=('interpolate',))
def maximum_likelihood(lnl, interpolate=DEFAULT_INTERPOLATE_BESTFIT):
    """Return maximum likelihood, and index of best-fit hypothesis
    
    Returns a tuple (y, i) of (n_trials, n_hypothesis) arrays, where y is the
        bestfit likelihood ratio and i the index of best-fit hypothesis.

    Arguments:
        lnl: (n_trials, n_hypotheses) array of log likelihoods
        interpolate: If True (default), the bestfit log likelihood is 
            interpolated parabolically when not at the edge of the likelihood 
            curve. The returned index i remains an interger.
    """
    n_outcomes, n_hyp = lnl.shape
    i = jnp.argmax(lnl, axis=-1)
    outcomes_is = jnp.arange(n_outcomes)
    y = lnl[outcomes_is, i]

    if not interpolate:
        return y, i
    
    # Estimate where the likelihood curve has zero gradient near its maximum
    _i_nearby = i[:,None] + jnp.array([-1, 0, 1])[None,:]
    lnl_grad = jnp.gradient(lnl, axis=-1)
    hyps_i = np.arange(n_hyp)
    # find_root_vec expects increasing y, but the gradient is decreasing
    # near the maximum -> use the negative gradient
    i_itp = nafi.find_root_vec(x=hyps_i, y=-lnl_grad, y0=0, i=_i_nearby)
    i_int, i_mod = jnp.floor(i_itp).astype(int), i_itp % 1
    y_itp = (
        lnl[outcomes_is,i_int] 
        # Note 0.5 from d[a x^2]/dx = 2 a x
        + 0.5 * i_mod * lnl_grad[outcomes_is, i_int])

    # Return interpolation only if the original i was not at the edge of the
    # likelihood curve (in which case points left or right of it are missing)
    allow_itp = (i > 0) & (i < n_hyp - 1)
    y = jnp.where(allow_itp, y_itp, y)

    return y, i


@export
@partial(jax.jit, static_argnames=('statistic', 'interpolate_bestfit'))
def test_statistics(
        lnl, 
        statistic=DEFAULT_TEST_STATISTIC, 
        interpolate_bestfit=DEFAULT_INTERPOLATE_BESTFIT):
    """Return test statistic computed from likelihood curves.

    The following statistics are supported:
    
      - ``t``: ``-2ln[ L(hypothesis)/L(bestfit) ]``
      - ``q``: as t, but zero if hypothesis <= bestfit (data is an excess), 
      - ``q0``: ``-2ln[ L(bestfit)/L(0) ]``, where '0' is the first hypothesis
      - ``lep``: ``-2ln[ L(hypothesis)/L(0) ]``
      - ``signedt``: as t, but -t if hypothesis <= bestfit (data is an excess)
      - ``signedtroot``: sqrt(t), but -sqrt(t) if hypothesis <= bestfit

    Arguments:
        lnl: (n_trials, n_hypotheses) array of log likelihoods
        statistic: name of the test statistic to use.
        interpolate_bestfit: If True, use interpolation to estimate 
            the best-fit hypothesis more precisely.

    Returns:
        Array of test statistics, same shape as lnl
    """
    _, n_hyp = lnl.shape

    lnl_best, best_i = maximum_likelihood(lnl, interpolate=interpolate_bestfit)

    # Compute statistics (|n_trials|,|n_hyps|)
    if statistic in ('t', 'q', 'signedt', 'signedtroot'):
        ts = -2 * (lnl - lnl_best[...,None])
        if statistic in ('q', 'signedt', 'signedtroot'):
            is_excess = jnp.arange(n_hyp)[None,:] <= best_i[...,None]
            if statistic == 'q':
                ts = jnp.where(is_excess, 0, ts)
            else:
                ts = jnp.where(is_excess, -ts, ts)
                if statistic == 'signedtroot':
                    ts = jnp.sqrt(ts)
    elif statistic == 'q0':
        # L(best)/L(0). Note this does not depend on the hypothesis.
        # Assuming hypothesis 0 is background only!
        ts = -2 * (lnl_best[...,None] - lnl[...,0][...,None])
        ts = ts.repeat(n_hyp, axis=-1)
    elif statistic == 'lep':
        # L(test)/L(0)
        ts = -2 * (lnl - lnl[...,0][...,None])
    else:
        raise ValueError(f"Unknown statistic {statistic}")
    return ts


@export
@partial(jax.jit, static_argnames=('statistic', 'cls'))
def asymptotic_pvals(ts, statistic=DEFAULT_TEST_STATISTIC, cls=DEFAULT_CLS):
    """Compute asymptotic frequentist right-tailed p-value for 
    test statistics ts.
    
    Arguments:
        ts: array of test statistics
        statistic: test statistic used; only 't' or 'q' implemented currently.
        cls: If True, use asymptotic formulae appropriate for the CLs method 
            instead of those for a frequentist/Neyman construction.
            (Currently this just raises NotImplementedError.)
    """
    # Compute asymptotic p-values
    if cls:
        raise NotImplementedError(
            "CLs asymptotics... might as well join LHC at this point")
    if statistic in ('t', 'q'):
        # The special function scipy uses here is quite expensive.
        ps = 1 - jax.scipy.stats.chi2.cdf(ts, df=1)
        if statistic == 'q':
            # q's distribution has a delta function at 0.
            ps = jnp.where(ts == 0, 1, 0.5 * ps)
    else:
        raise NotImplementedError(
            "No asymptotic distribution implemented for statistic {statistic}")
    return ps


# Bit of a hack to use static_argnames for freeze_truth_index below;
# if people try varying the frozen index in some kind of loop, it will trigger
# unexpected recompilation. But for CLs we just always freeze to one hypothesis;

@export
@partial(jax.jit, static_argnames=('freeze_truth_index'))
def neyman_pvals(ts, toy_weight, freeze_truth_index=None):
    """Compute right-tailed p-values from test statistics ts using a 
    Neyman construction.
    
    Arguments:
        ts: array of test statistics, shape (n_trials, n_hypotheses)
        toy_weight: array of toy outcome weights, same shape
        freeze_truth_index: If set, freezes the toy weights to the given 
            hypothesis index. Useful for CLs.

    """
    return neyman_pvals_from_ordering(
        *nafi.order_and_index(ts), toy_weight, 
        freeze_truth_index=freeze_truth_index)


@partial(jax.jit, static_argnames=('freeze_truth_index'))
def neyman_pvals_from_ordering(
        order, sort_index, toy_weight, 
        freeze_truth_index=None):
    """Return right-tailed p-values of outcomes given an hypothesis-dependent
    ordering of outcomes.

    Arguments:
        order, sort_index: ordering of outcomes, from nafi.order_and_index.
            Both are (n_outcomes, n_hypotheses) arrays.
        toy_weight: array of toy outcome weights, same shape.
        freeze_truth_index: If set, freezes the toy weights to the given 
            hypothesis index. Useful for CLs.
    """
    # For some reason, vmap over in_axis=0 is much faster than in_axis=1 and 
    # transposing the inputs... hm....
    if freeze_truth_index is not None:
        # Always evaluate with _one_ true hypothesis
        toy_weight = toy_weight[:,freeze_truth_index]
        in_axes = (0, 0, None)
    else:
        # Compute weighted_ps independently for each hypothesis
        in_axes = (0, 0, 0)
        toy_weight = toy_weight.T
    return jax.vmap(nafi.utils._weighted_ps_presorted, in_axes=in_axes)(
        order.T, sort_index.T, toy_weight).T


# TODO: could we sensibly jit this even though it has a for loop?
@export
def neyman_pvals_weighted(
        ts, hypotheses, weight_function, 
        *outcomes, 
        progress=True, freeze_truth_index=None, 
        **parameters):
    """Compute right-tailed p-values from test statistics ts using a 
    Neyman construction, where hypothetical outcomes are weighted by a
    function of the observed outcome. 
    This is used for the profile construction.
    
    Arguments:
      ts: test statistic, shape (n_outcomes, n_hypotheses)
      hypotheses: hypotheses, shape (n_hypotheses,)
      weight_function: function that returns weights of hypothetical outcomes
        given an actually observed outcome. Called as:
        ``weight_function(hypotheses, *outcomes, *observed_outcome, **parameters)``
      outcomes: one or more (n_outcomes,) arrays of possible outcomes.
           E.g. (n, m) arrays for a two-bin experiment.
      progress: whether to show a progress bar
      parameters: additional parameters to pass to weight_function
    """
    # Find the sort order of outcomes by test statistic, for each hypothesis.
    # (Should do this now rather than in the loop over outcomes below!)
    order, sort_index = nafi.order_and_index(ts)

    # Would love to do this in jax, but we get out of memory errors
    # (probably due to the huge reduction at the end of p_given_obs)
    # ps_2 = jax.vmap(p_2, in_axes=(None, None, None, None, None, None, 0))(
    #     order, sort_index, mu_sig, tau, n, m, jnp.arange(n.size))
    return np.stack([
        _profile_p_given_obs(
            weight_function,
            order, sort_index, 
            hypotheses, 
            *outcomes, 
            outcome_i=i, 
            freeze_truth_index=freeze_truth_index,
            **parameters)
        for i in nafi.utils.tqdm_maybe(progress)(range(outcomes[0].size))])


@partial(jax.jit, static_argnames=('weight_function', 'freeze_truth_index',))
def _profile_p_given_obs(
        weight_function,
        order, sort_index, 
        hypotheses, 
        *outcomes, 
        outcome_i, 
        freeze_truth_index, 
        **parameters):
    """Return p-value of the outcome with index outcome_i, weighting 
    hypothetical outcomes with a weight_function that depends on the actually
    observed outcome (as in the profile construction).
    """
    # Weights of all hypothetical outcomes
    weights = weight_function(
        hypotheses, 
        *outcomes,                           # Possible outcome
        *[r[outcome_i] for r in outcomes],   # Actually observed outcome
        **parameters)
    # P-values of all outcomes
    ps = neyman_pvals_from_ordering(
        order, sort_index, weights,
        freeze_truth_index=freeze_truth_index)
    # Only care about the observed outcome..
    return ps[outcome_i]


@export
def cls_pvals(ts, toy_weight, neyman_ps=None):
    """Compute p-value ratios used in the CLs method. 
    First hypothesis must be background-only.
    """
    order, sort_index = nafi.order_and_index(ts)
    if neyman_ps is None:
        neyman_ps = neyman_pvals_from_ordering(
            order, sort_index, toy_weight)
    ps_0 = neyman_pvals_from_ordering(
        order, sort_index, toy_weight, freeze_truth_index=0)

    # PDG review and other CLs texts have a 1- in the denominator
    # because they define "p_b" to be a CDF value (integrate distribution 
    # from -inf to the observed value), unlike "p_{s+b}".
    # Our ps are both survival function values (integrate distribution from
    # observed value to +inf)

    # This definition assumes the statistics are like q_mu, i.e.
    # decreasing if we add more events. If using a statistic with the opposite
    # behaviour (as older CLs papers do), we would need left-tailed p-values. 
    # (And left- and right-tailed p-values are not just 1 - complements of each
    #  other, since multiple outcomes may achieve the same test statistic.)
    return neyman_ps / ps_0


# Shortcut API that combines the above functions
@export
def ts_and_pvals(
        lnl, 
        toy_weight, 
        statistic=DEFAULT_TEST_STATISTIC, 
        cls=DEFAULT_CLS,
        interpolate_bestfit=DEFAULT_INTERPOLATE_BESTFIT, 
        asymptotic=False):
    """Compute frequentist test statistics and p-values for likelihood ratios
    
    Parameters
    ----------
    lnl : array
        Likelihood ratio, shape (n_trials, n_hypotheses)
    toy_weight : array
        (Hypothesis-dependent) weight of each toy, shape (trials, n_hyp)
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
    ts = test_statistics(lnl, statistic, interpolate_bestfit=interpolate_bestfit)

    # Convert stats to p-values
    if asymptotic:
        ps = asymptotic_pvals(ts, statistic, cls=cls)
    elif cls:
        ps = cls_pvals(ts, toy_weight)
    else:
        ps = neyman_pvals(ts, toy_weight)
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

    Returns a tuple (ts, ps), where ts is an (n_hypotheses,) array of test
        statistic values at the hypotheses, and ps the same array of p-values.
    
    Arguments:
      lnl_obs: (n_hypotheses,) array of log likelihoods for one trial
      ts: (n_toys, n_hypotheses,) array of toy test statistics
      ps: (n_toys, n_hypotheses,) array of toy p-values
      asymptotic: If True, use asymptotic distribution of the test statistic
      asymptotic_cls: If True, CLs asymptotics are used if asymptotic=True.
        (Argument is not called 'cls' to avoid people feeding in Neyman p-values
        and thinking setting this gets them Cls results.)
    """
    # Compute test statistic
    # shape: (1, |n_hypotheses|)
    t_obs = test_statistics(
        lnl_obs[None,:], 
        statistic=statistic, 
        interpolate_bestfit=interpolate_bestfit)
    if asymptotic:
        p_obs = asymptotic_pvals(t_obs, statistic=statistic, cls=asymptotic_cls)
    else:
        assert ts is not None and ps is not None, "Need toy test statistics and p-values"
        # Get index of toy trial with the same (or a very close) test statistic
        # (for each hypothesis)
        # shape: (|n_hypotheses|)
        closest_trial = np.argmin(np.abs(ts - t_obs), axis=0)
        # Get corresponding p-value from ps
        # shape: (|n_hypotheses|)
        p_obs = ps[closest_trial, np.arange(len(closest_trial))]
    return t_obs[:,0], p_obs
