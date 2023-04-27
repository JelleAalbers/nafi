import jax
import jax.numpy as jnp
import numpy as np

import nafi
export, __all__ = nafi.exporter()


SIGMAS = np.array([-2, -1, 0, 1, 2])


@export
def outcome_probabilities(ll, ul, toy_weight, hypotheses):
    """Returns dict with probabilities (n_hypotheses arrays) of:
        - mistake: false exclusion of the hypotheses when it is true
        - mistake_ul: same, counting only exclusions by the upper limit
        - empty: empty confidence interval
        - discovery: exclusion of hypothesis 0, and interval not empty
        - bg_exclusion: exclusion of hypothesis, when hypothesis 0 is true
        - bg_exclusion_ul: same, counting only exclusions by the upper limit
    """
    assert np.all(ll <= ul), "First argument must be the lower limits"
    empty_interval = np.isnan(ul) & np.isnan(ll)

    def get_p(bools):
        if len(bools.shape) == 1:
            # Things like empty interval don't depend on the hypothesis,
            # except through weighting of the toys
            bools = bools[:,None]
        return np.sum(bools * toy_weight, axis=0)

    # Compute P(mu excluded | mu is true) -- i.e. coverage
    ul_is_ok = hypotheses[None,:] <= ul[:,None]
    ll_is_ok = ll[:,None] <= hypotheses[None,:]
    p_mistake = 1 - get_p(ul_is_ok & ll_is_ok)
    p_false_ul = 1 - get_p(ul_is_ok)

    # Compute P(mu excluded | 0 is true)
    # TODO: this makes (n_trials, n_hyp) arrays again
    # is there a more memory-efficient solution?
    ll_allows_mu = ll[:,None] <= hypotheses[None,:]
    ul_allows_mu = hypotheses[None,:] <= ul[:,None]
    p_excl_bg = 1 - np.average(
        ll_allows_mu & ul_allows_mu, weights=toy_weight[:,0], axis=0)
    p_excl_bg_ul = 1 - np.average(
        ul_allows_mu, weights=toy_weight[:,0], axis=0)

    # Compute P(0 excluded | mu is true) -- only need lower limits
    excludes_zero = ll > 0

    # Compute P(empty interval | mu is true)
    p_empty = get_p(empty_interval)
    
    # Compute P(discovery | mu is true)
    p_disc = get_p(excludes_zero & (~empty_interval))

    return dict(
        mistake=p_mistake,
        mistake_ul=p_false_ul,
        empty=p_empty,
        discovery=p_disc,
        bg_exclusion=p_excl_bg,
        bg_exclusion_ul=p_excl_bg_ul)


@export
def brazil_band(limits, toy_weight, return_array=False):
    """Return a dictionary with "Brazil band" quantiles.
    Keys are the five sigma levels, e.g. -1 gives the -1σ quantile 
    (the 15.8th percentile).

    Arguments:
     - limits: Upper (or lower) limits, shape (n_trials,)
     - toy_weight: weights of the toys, shape (n_trials, n_hypotheses)
     - return_array: if False, instead returns a (n_hyp, 5) array,
       with the second axis running over levels (index 0 = -2σ, 1 = -1σ, etc)
     - progress: whether to show a progress bar
    """
    sensi = _brazil_band(limits, toy_weight, SIGMAS)
    if return_array:
        return sensi
    return {sigma: sensi for sigma, sensi in zip(SIGMAS.tolist(), sensi.T)}


@jax.jit
def _brazil_band(limits, toy_weight, sigmas):
    quantiles = jax.scipy.stats.norm.cdf(sigmas)

    # Sort once, we need the same order every hypothesis in the vmap
    sort_order = jnp.argsort(limits)
    sorted_limits = limits[sort_order]
    sorted_weights = toy_weight[sort_order, :]

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
