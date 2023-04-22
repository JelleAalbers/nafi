import numpy as np

from scipy import stats

import nafi
export, __all__ = nafi.exporter()


@export
def outcome_probabilities(ul, ll, toy_weight, hypotheses):
    """Returns dict with probabilities (n_hypotheses arrays) of:
        - mistake: false exclusion of the hypotheses when it is true
        - mistake_ul: same, counting only exclusions by the upper limit
        - empty: empty confidence interval
        - discovery: exclusion of hypothesis 0, and interval not empty
        - bg_exclusion: exclusion of hypothesis, when hypothesis 0 is true
        - bg_exclusion_ul: same, counting only exclusions by the upper limit
    """
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
def brazil_band(ul, ll, toy_weight, progress=False, return_dict=True):
    """Return sensitivity quantiles for the upper and lower limits.

    Arguments:
     - ul, ll: upper and lower limits of the confidence intervals
     - toy_weight: weights of the toys
     - progress: whether to show a progress bar
     - return_dict: if True (default), returns two dict with keys -2, -1, 0, 1, 2
        containing the sensitivity quantiles at -2 sigma, -1 sigma, etc.
        Otherwise returns a (n_hyp, 5) array, with the second axis containing
        the same quantiles in the same order.

    """
    n_hyp = toy_weight.shape[-1]
    sigmas = np.array([-2, -1, 0, 1, 2])
    n_sigmas = len(sigmas)
    quantiles = stats.norm.cdf(sigmas)
    sensi_ll = np.zeros((n_hyp, n_sigmas))
    sensi_ul = np.zeros((n_hyp, n_sigmas))
    for mu_i in nafi.utils.tqdm_maybe(progress)(
            range(n_hyp), desc='Computing sensitivity quantiles', leave=False):
        weights = toy_weight[...,mu_i].ravel()
        sensi_ul[mu_i] = nafi.utils.weighted_quantile(
            values=ul.ravel(), quantiles=quantiles, weights=weights)
        sensi_ll[mu_i] = nafi.utils.weighted_quantile(
            values=ll.ravel(), quantiles=quantiles, weights=weights)
    if not return_dict:
        return sensi_ll, sensi_ul
    # TODO: deduplicate
    sensi_ll = {sigma: sensi for sigma, sensi in zip(sigmas, sensi_ll.T)}
    sensi_ul = {sigma: sensi for sigma, sensi in zip(sigmas, sensi_ul.T)}
    return sensi_ll, sensi_ul
    
