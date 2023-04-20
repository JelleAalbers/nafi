import numpy as np

from scipy import stats

import nafi
export, __all__ = nafi.exporter()


@export
def outcome_probabilities(ul, ll, hypotheses, p_n_mu=None):
    """Returns dict with probabilities (n_hypotheses arrays) of:
        - mistake: false exclusion of the hypotheses when it is true
        - mistake_ul: same, counting only exclusions by the upper limit
        - empty: empty confidence interval
        - discovery: exclusion of hypothesis 0, and interval not empty
        - bg_exclusion: exclusion of hypothesis, when hypothesis 0 is true
        - bg_exclusion_ul: same, counting only exclusions by the upper limit
    """
    toy_weight = nafi.utils.toy_weights(
        shape=ul.shape,
        p_n_mu=p_n_mu, hypotheses=hypotheses)
    empty_interval = np.isnan(ul) & np.isnan(ll)

    def get_p(bools):
        if len(bools.shape) == 2:
            # Things like empty interval don't depend on the hypothesis,
            # except through weighting of the toys
            bools = bools[:, :, None]
        return np.sum(bools * toy_weight, axis=(0, 1))

    # Compute P(mu excluded | mu is true) -- i.e. coverage
    ul_is_ok = hypotheses[None,None,:] <= ul[:,:,None]
    ll_is_ok = ll[:,:,None] <= hypotheses[None,None,:]
    p_mistake = 1 - get_p(ul_is_ok & ll_is_ok)
    p_false_ul = 1 - get_p(ul_is_ok)

    # Compute P(mu excluded | 0 is true) -- only need the bg-only toys
    ll_0, ul_0 = ll[:,0], ul[:,0]
    # These two are (n_hypotheses, n_bg_toys) arrays
    ll0_allows_mu = ll_0[None,:] <= hypotheses[:,None]
    ul0_allows_mu = hypotheses[:,None] <= ul_0[None,:]

    p_excl_bg = 1 - np.mean(ll0_allows_mu & ul0_allows_mu, axis=1)
    p_excl_bg_ul = 1 - np.mean(ul0_allows_mu, axis=1)

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
def brazil_band(ul, ll, hypotheses=None, p_n_mu=None, progress=False):
    toy_weight = nafi.utils.toy_weights(
        shape=ul.shape, p_n_mu=p_n_mu, hypotheses=hypotheses)
    sigmas = np.array([-2, -1, 0, 1, 2])
    n_sigmas = len(sigmas)
    quantiles = stats.norm.cdf(sigmas)
    sensi_ll = np.zeros((len(hypotheses), n_sigmas))
    sensi_ul = np.zeros((len(hypotheses), n_sigmas))
    for mu_i in nafi.utils.tqdm_maybe(
            range(len(hypotheses)), desc='Computing sensitivity quantiles', leave=False):
        weights = toy_weight[...,mu_i].ravel()
        sensi_ul[mu_i] = nafi.utils.weighted_quantile(values=ul.ravel(), quantiles=quantiles, weights=weights)
        sensi_ll[mu_i] = nafi.utils.weighted_quantile(values=ll.ravel(), quantiles=quantiles, weights=weights)
