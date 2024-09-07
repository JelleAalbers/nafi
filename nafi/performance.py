from functools import partial
import jax
import jax.numpy as jnp

import nafi
export, __all__ = nafi.exporter()


SIGMAS = jnp.array([-2, -1, 0, 1, 2])



@export
@partial(jax.jit, static_argnames=('singular_is_empty'))
def outcome_probabilities(ll, ul, weights, hypotheses,
                          singular_is_empty=False):
    """Returns probabilities that confidence intervals satisfy some
    properties.

    Dictionary keys are as follows, with values (n_hypotheses,) arrays:
    -  ``mistake``: false exclusion of the hypotheses when it is true
    -  ``mistake_ul``: same, considering only exclusions by the upper limit
    -  ``degenerate``: empty interval
    -  ``discovery``: exclusion of hypothesis 0, and interval not degenerate
    -  ``excl_if_bg``: this hypothesis excluded when hypothesis 0 is true
    -  ``excl_if_bg_ul``: same, counting only exclusions by the upper limit
    -  ``excl_if_bg_ul``: same, counting only exclusions by the lower limit

    Arguments:
      ll: (outcomes,) array of lower limits
      ul: (outcomes,) array of upper limits
      weights: (outcomes, hypotheses) array of normalized P(outcome|hypothesis)
      hypotheses: (hypotheses,) array of hypotheses
      singular_is_empty: If True, consider single-hypothesis
        intervals as empty intervals. Otherwise only truly empty intervals
        are considered empty.

    """
    if singular_is_empty:
        is_singular = (ul == ll)
        ul = jnp.where(is_singular, jnp.nan, ul)
        ll = jnp.where(is_singular, jnp.nan, ll)

    # Compute P(degenerate interval | mu is true)
    empty = jnp.isnan(ul) & jnp.isnan(ll)
    p_empty = _get_p(empty, weights)

    # Compute P(mu excluded | mu is true) -- i.e. coverage
    ul_is_ok = hypotheses[None,:] <= ul[:,None]
    ll_is_ok = ll[:,None] <= hypotheses[None,:]
    p_mistake = 1 - _get_p(ul_is_ok & ll_is_ok, weights)
    p_false_ul = 1 - _get_p(ul_is_ok, weights)
    p_false_ll = 1 - _get_p(ll_is_ok, weights)

    # Compute P(mu excluded | 0 is true)
    # TODO: this makes (n_trials, n_hyp) arrays again
    # is there a more memory-efficient solution?
    ll_allows_mu = ll[:,None] <= hypotheses[None,:]
    ul_allows_mu = hypotheses[None,:] <= ul[:,None]
    p_excl_bg = 1 - jnp.average(
        ll_allows_mu & ul_allows_mu, weights=weights[:,0], axis=0)
    # Compute P(mu excluded by UL | 0 is true)
    # The inverse of this maps quantiles to the "Brazil band"
    p_excl_bg_ul = 1 - jnp.average(
        ul_allows_mu, weights=weights[:,0], axis=0)
    p_excl_bg_ll = 1 - jnp.average(
        ll_allows_mu, weights=weights[:,0], axis=0)

    # P(discovery | mu is true)
    # here discovery = lower limit > 0 (and interval nonempty)
    ll_liftoff = (ll > hypotheses[0])
    p_disc = _get_p(ll_liftoff, weights)

    return dict(
        mistake=p_mistake,
        mistake_ul=p_false_ul,
        mistake_ll=p_false_ll,
        degenerate=p_empty,
        discovery=p_disc,
        excl_if_bg=p_excl_bg,
        excl_if_bg_ul=p_excl_bg_ul,
        excl_if_bg_ll=p_excl_bg_ll)


def _get_p(bools, weights):
    """Return P(condition | hypothesis)

    Arguments:
      bools: (outcome, hyp) or (outcome,) array
      weights: (outcomes, hypotheses) array of normalized P(outcome|hypothesis)
    """
    if len(bools.shape) == 1:
        # bools is a (outcome,) array
        # I.e. a condition that does not depend on the hypothesis
        # (except through weighting of the toys)
        bools = bools[:,None]
    # Floating-point errors sometimes cause negativate values.
    # TODO: Does this only happen for CLs?
    return jnp.sum(bools * weights, axis=0).clip(0,1)


@export
@partial(jax.jit, static_argnames=('singular_is_empty'))
def twod_power(
        ll, ul, weights, hypotheses,
        singular_is_empty=False):
    """Returns (hypothesis, truth) square array with P(hyp excluded | truth)

    Arguments:
      ll: (outcomes,) array of lower limits
      ul: (outcomes,) array of upper limits
      weights:(outcomes, hypotheses) array of normalized P(outcome|hypothesis)
      hypotheses: (hypotheses,) array of hypotheses
      singular_is_empty: If True, consider single-hypothesis
        intervals as empty intervals. Otherwise only truly empty intervals
        are considered empty.
    """
    if singular_is_empty:
        is_singular = (ul == ll)
        ul = jnp.where(is_singular, jnp.nan, ul)
        ll = jnp.where(is_singular, jnp.nan, ll)

    # (outcome, hyp) arrays: does interval from outcome allow the hypothesis?
    # NaNs will give False in these comparisons.
    is_allowed = (
        (ll[:,None] <= hypotheses[None,:])
        & (hypotheses[None,:] <= ul[:,None]))

    return 1 - jax.vmap(_get_p, in_axes=(1, None))(is_allowed, weights)


@export
def brazil_band(limits, weights, return_array=False):
    """Return a dictionary with "Brazil band" quantiles.
    Keys are the five sigma levels, e.g. -1 gives the -1σ quantile
    (the 15.8th percentile).

    Arguments:
      limits: Upper (or lower) limits, shape (n_trials,)
      weights: (outcomes, hypotheses) array of normalized P(outcome|hypothesis)
      return_array: if False, instead returns a (n_hyp, 5) array,
        with the second axis running over levels (index 0 = -2σ, 1 = -1σ, etc)
      progress: whether to show a progress bar
    """
    sensi = _brazil_band(limits, weights, SIGMAS)
    if return_array:
        return sensi
    return {sigma: sensi for sigma, sensi in zip(SIGMAS.tolist(), sensi.T)}


@jax.jit
def _brazil_band(limits, weights, sigmas):
    quantiles = jax.scipy.stats.norm.cdf(sigmas)

    # Sort once, we need the same order every hypothesis in the vmap
    sort_order = jnp.argsort(limits)
    sorted_limits = limits[sort_order]
    sorted_weights = weights[sort_order, :]

    # vmap over hypotheses
    sensi = jax.vmap(
        nafi.utils.weighted_quantile_sorted,
        in_axes=(None, 1, None),
        )(
            sorted_limits,
            sorted_weights,
            quantiles,
        )
    return sensi


@export
@jax.jit
def credibility(posterior_cdf, hypotheses, ll, ul):
    """Return Bayesian probability content of intervals [ll, ul]
    given a posterior CDF.

    As in posterior_cdf, the hypotheses are assumed to be merely discrete
    representatives from an underlying continuous parameter that is of interest.

    Arguments:
      posterior_cdf: posterior, shape (n_outcomes, n_hypotheses,)
        See nafi.posterior_cdf, don't just cumsum your posterior if you care
        about off-by-half errors.
      hypotheses: hypotheses, shape (n_hypotheses,)
      ll: lower limits, shape (n_outcomes,). If omitted, uses hypotheses[0].
      ul: upper limits, shape (n_outcomes,). If omitted, uses hypotheses[-1].
    """
    # P(truth <= h) = P(truth < h), see posterior_cdf
    hyp_to_p = jax.vmap(jnp.interp, in_axes=(0, None, 0))
    cred = (
        hyp_to_p(ul, hypotheses, posterior_cdf)
        - hyp_to_p(ll, hypotheses, posterior_cdf)
        # ... no need to add P(truth = ll), parameter is continuous so
        # that is zero
    )
    return cred
