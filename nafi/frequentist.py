"""Frequentist methods for computing p-values from likelihoods
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import nafi

export, __all__ = nafi.exporter()

DEFAULT_TEST_STATISTIC = 't'
DEFAULT_INTERPOLATE_BESTFIT = False
DEFAULT_CLS = False


@jax.jit
def _parabola_vertex(x1, y1, x2, y2, x3, y3):
    # From https://stackoverflow.com/questions/717762
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2))
    B     = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3))
    C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3)
    return -B / (2*A), (C - B*B / (4*A))/denom


@export
@partial(jax.jit, static_argnames=('interpolate',))
def maximum_likelihood(lnl, interpolate=DEFAULT_INTERPOLATE_BESTFIT):
    """Return maximum likelihood, and index of best-fit hypothesis
    
    Returns two (|trials|, |n_sig|) arrays: the bestfit likelihood ratio and
    index of best-fit hypothesis.

    If interpolate=True, bestfit likelihood is interpolated parabolically
        when it is not at the edge of the likelihood curve.
        The index will be a float in this case.
    """
    n_outcomes, n_hyp = lnl.shape
    i = jnp.argmax(lnl, axis=-1)
    y = lnl[jnp.arange(n_outcomes), i]

    if not interpolate:
        return y, i
    
    # TODO: interpolating the best fit has a bug!
    # Need unit tests for this..

    # Jax auto-clamps indices, so this won't crash
    y_prev = lnl[jnp.arange(n_outcomes), i - 1]
    y_next = lnl[jnp.arange(n_outcomes), i + 1]
    # jax.debug.print(
    #     "i-1 = {0}\ny_prev = {1}\ni = {2}\ny = {3}\ni+1 = {4}\ny_next = {5}",
    #     i[14:19] - 1, 
    #     y_prev[14:19],
    #     i[14:19], 
    #     y[14:19], 
    #     i[14:19] + 1, 
    #     y_next[14:19])
    i_itp, y_itp = _parabola_vertex(i - 1, y_prev, i, y, i + 1, y_next)
    # Return interpolated solution only if all the following hold:
    #   * The original i was not at the edge of the likelihood curve
    #     (in which case points left or right of it are missing)
    #   * i_itp is within [i-1, i+1], i.e. we are not extrapolating
    #   * y_itp is higher than y (it improves the fit)
    #   * y_prev and y_next are both lower than y by a reasonable amount
    #     to exclude numerical errors (since we usually use float32)
    # The last two shouldn't really be necessary if the interpolation is
    # working correctly, but I don't know what kinds of mad likelihoods
    # people will throw at this...
    allow_itp = (
        (i > 0) & (i < n_hyp - 1)
        & (jnp.abs(i_itp - i) <= 1)
        & (y_itp > y)
        & (jnp.minimum(y_prev, y_next) < 0.9 * (y_itp - 1e-1))
        )
    y = jnp.where(allow_itp, y_itp, y)
    i = jnp.where(allow_itp, i_itp, i)
    return y, i


@export
@partial(jax.jit, static_argnames=('statistic', 'interpolate_bestfit'))
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
            * signedt is t, but -t if hypothesis < bestfit (data is an excess)
        interpolate_bestfit: If True, use interpolation to estimate 
            the best-fit hypothesis more precisely.

    Returns:
        Array of test statistics, same shape as lnl
    """
    _, n_hyp = lnl.shape

    lnl_best, best_i = maximum_likelihood(lnl, interpolate=interpolate_bestfit)

    # Compute statistics (|n_trials|,|n_hyps|)
    if statistic in ('t', 'q', 'signedt'):
        ts = -2 * (lnl - lnl_best[...,None])
        if statistic in ('q', 'signedt'):
            is_excess = jnp.arange(n_hyp)[None,:] <= best_i[...,None]
            if statistic == 'q':
                ts = jnp.where(is_excess, 0, ts)
            else:
                ts = jnp.where(is_excess, -ts, ts)
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
    ps = 1 - jax.scipy.stats.chi2.cdf(ts, df=1)
    if statistic == 'q':
        ps *= 0.5
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
      - ts: test statistic, shape (n_outcomes, n_hypotheses)
      - hypotheses: hypotheses, shape (n_hypotheses,)
      - weight_function: function that returns weights of hypothetical outcomes
        given an actually observed outcome. Called as:
        weight_function(hypotheses, *outcomes, *observed_outcome, **parameters)
      - outcomes: one or more (n_outcomes,) arrays of possible outcomes.
            E.g. (n, m) arrays for a two-bin experiment.
      - progress: whether to show a progress bar
      - parameters: additional parameters to pass to weight_function
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


# Higher level API, combining the above functions
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
        # Get index of toy trial with the same (or a very close) test statistic
        # (for each hypothesis)
        # shape: (|n_hypotheses|)
        closest_trial = np.argmin(np.abs(ts - t_obs), axis=0)
        # Get corresponding p-value from ps
        # shape: (|n_hypotheses|)
        p_obs = ps[closest_trial, np.arange(len(closest_trial))]
    return t_obs[:,0], p_obs
