import nafi
import numpy as np

from nafi.likelihoods import counting

def test_synopsis():
    # Test a coarser hypothesis grid, so the test runs quickly
    hypotheses = np.arange(0, 42, 0.1)

    mu_bg = 10
    lnl, weights = counting.lnl_and_weights(mu_sig=hypotheses, mu_bg=mu_bg)
    ts, ps = nafi.ts_and_pvals(lnl, weights)
    _, upper_limits = nafi.intervals(ps, hypotheses, cl=0.9)
    brazil = nafi.brazil_band(upper_limits, weights)
    lnl_obs = counting.single_lnl(n=17, mu_sig=hypotheses, mu_bg=mu_bg)
    _, p_obs = nafi.single_ts_and_pvals(lnl_obs, ts=ts, ps=ps)
    ll_obs, ul_obs = nafi.intervals(p_obs, hypotheses, cl=0.9)

    # Check results remain those quoted in the README
    # (up to some lower accuracy, we only sdid 0.1 hypothesis spacing here)
    np.testing.assert_almost_equal(round(p_obs[0], 4), 0.0270)
    np.testing.assert_almost_equal(round(ll_obs, 1), 1.8, decimal=3)
    np.testing.assert_almost_equal(round(ul_obs, 1), 15.1, decimal=3)
    np.testing.assert_almost_equal(round(brazil[0][0], 1), 6.6, decimal=3)
    np.testing.assert_almost_equal(round(brazil[-1][0], 1), 3.1, decimal=3)
    np.testing.assert_almost_equal(round(brazil[+1][0], 1), 10.1, decimal=3)
