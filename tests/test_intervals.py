import nafi
import numpy as np

from nafi.likelihoods import counting


def test_intervals():
    cl = 0.9

    # Counting experiment with bg = 10
    # use CLs with t, which gives horrible nonsense as p-values
    # TODO: instead we could random p-values with the 'hypothesis' pypi package
    # (but would have to make test below robust to empty intervals)

    hypotheses = np.arange(0, 42, 0.1)
    mu_bg = 10
    lnl, weights = counting.lnl_and_weights(mu_sig=hypotheses, mu_bg=mu_bg)
    _, ps = nafi.ts_and_pvals(lnl, weights, cls=True)
    allowed = ps >= (1 - cl)

    # (n_outcomes,) arrays
    for interpolate in False, True:
        ll, ul = nafi.intervals(ps, hypotheses, cl=cl, interpolate=interpolate)

        # Intervals allow more than one hypothesis
        # NB: this will fail on a problem with empty intervals
        assert np.all(ul > ll)

        # Test [lower limit is 0] === [mu=0] is allowed

        # lower limit is 0 if mu=0 is allowed 
        assert np.all((ll == 0) | (~allowed[:,0]))

        # lower limit is >0 if mu=0 is excluded,
        # NB: this will fail on a problem with empty intervals
        assert np.all((ll > 0) | (allowed[:,0]))

        # Test we get empty intervals (represented by NaNs) if all ps < alpha
        ps_all_bad = np.zeros_like(ps) + (1 - cl)/2
        ll, ul = nafi.intervals(ps_all_bad, hypotheses, cl=cl, interpolate=interpolate)
        assert np.all(np.isnan(ll) & np.isnan(ul))

        # Test we get full intervals if all ps >= alpha
        ps_all_good = np.ones_like(ps) + (1 - cl)
        ll, ul = nafi.intervals(ps_all_good, hypotheses, cl=cl, interpolate=interpolate)
        assert np.all(ll == hypotheses[0])
        assert np.all(ul == hypotheses[-1])
