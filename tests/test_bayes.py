import nafi

import numpy as np


def test_bayesian_intervals():
    dmu = 0.01
    mu_max = 42
    hypotheses = np.arange(0, mu_max, dmu)

    # Counting experiment without background
    lnl, weights, n_obs = nafi.likelihoods.counting.lnl_and_weights(
        mu_sig=hypotheses, mu_bg=0, return_outcomes=True)
    valid = n_obs < mu_max / 2

    # Bayesian upper limits, uniform prior
    ps_bayes = nafi.bayesian_pvals(lnl, hypotheses, interval_type='ul')
    ll_bayes, ul_bayes = nafi.intervals(ps_bayes, hypotheses)
    np.testing.assert_allclose(ll_bayes, 0)

    # Bayesian and Frequentist upper limits should be identical for this case
    # (except near the upper limit of the hypotheses range, where things
    #  become invalid)
    ul_desired = nafi.poisson_ul(n_obs)
    np.testing.assert_allclose(
        ul_bayes[valid], 
        ul_desired[valid], 
        # Has to be less than dmu to catch off-by-one error
        # (e.g. from subtracting the posterior from the CDF
        #  to not include the p of the hypothesis itself)
        atol=dmu * 0.7)

    # Bayesian limits must have accurate credibility
    posterior = nafi.posterior(lnl, hypotheses)
    
    # Calculate credibility of the provided upper limit,
    # and of the upper limit plus/minus dmu
    cred_less, cred, cred_more = [
        nafi.credibility(
            nafi.posterior_cdf(posterior), 
            hypotheses, 
            ll=np.zeros(n_obs.size) + hypotheses[0], 
            ul=ul_bayes + dmu * i)
        for i in [-1, 0, 1]
    ]
    # Allowed deviation should be less than what results from
    # moving the UL by a single dmu, i.e.
    # (This actually also holds for hypotheses near the 
    #  'invalid' upper boundary.)
    allowed_diff = np.maximum(
        cred_more - cred,
        cred - cred_less)
    np.testing.assert_array_less(
        np.abs(cred - 0.9),
        # Factor has to be less than 1 to catch off-by-one error
        # and well below 1 to stress-test interpolation
        0.1 * allowed_diff,
    )

    ##
    # Test high-density posterior intervals
    ##
    posterior = nafi.posterior(lnl, hypotheses)
    ps_bayes = nafi.bayesian_pvals(lnl, hypotheses, interval_type='hdpi')
    ll_bayes, ul_bayes = nafi.intervals(ps_bayes, hypotheses)
    assert not np.all(ll_bayes == 0)

    # Calculate credibility of the provided results,
    # and of the lower limit plus/minus dmu
    cred_less, cred, cred_more = [
        nafi.credibility(
            nafi.posterior_cdf(posterior), 
            hypotheses, 
            ll=ll_bayes,
            ul=ul_bayes + dmu * i)
        for i in [-1, 0, 1]
    ]
    # Allowed deviation should be less than what results from
    # moving the UL by a single dmu, i.e.
    allowed_diff = np.maximum(
        cred_more - cred,
        cred - cred_less)
    np.testing.assert_array_less(
        np.abs(cred - 0.9)[valid],
        # Factor has to be less than 1 to catch off-by-one error
        # and well below 1 to stress-test interpolation
        0.3 * allowed_diff[valid],
    )
