import nafi

import numpy as np


def test_poisson_limit():
    dmu = 0.01
    mu_max = 42
    hypotheses = np.arange(0, mu_max, dmu)

    # Counting experiment without background
    lnl, weights, n_obs = nafi.likelihoods.counting.lnl_and_weights(
        mu_sig=hypotheses, mu_bg=0, return_outcomes=True)
    
    _, ps = nafi.ts_and_pvals(lnl, weights, cls=True, statistic='q')
    _, ul = nafi.intervals(ps, hypotheses)

    ul_desired = nafi.poisson_ul(n_obs)
    valid = n_obs < mu_max / 2
    np.testing.assert_allclose(
        ul[valid], 
        ul_desired[valid], 
        # Has to be less than dmu to catch off-by-one error
        # and well below dmu to stress-test interpolation
        atol=0.1 * dmu)


def test_maximum_likelihood():
    # Test we find the exact maximum likelihood if the likelihood
    # is parabolic (linear gradient)

    # Some random parabola; maximum is x= -2.5, y = 5.5
    def f(x):
        return - (0.4 * x**2 + 2 * x - 3)
    
    # Evaluate it only on a grid
    for delta in [0, 0.123, -0.123]:
        x = np.arange(-6, 6) + delta
        y = f(x)

        # Test parabolic interpolation is exact 
        assert np.allclose(
            nafi.maximum_likelihood(y[None,:], interpolate=True)[0], 
            5.5)

