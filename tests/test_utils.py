import numpy as np

import nafi


def test_weighted_quantiles():
    # Ensure quantiles are not interpolated
    # (quantiles should be real existing values in the dataset)
    assert 0 == nafi.weighted_quantile(
        values=np.array([0, 1]), 
        weights=np.array([0.6, 0.4]), 
        quantiles=0.5).item()
    assert 1 == nafi.weighted_quantile(
        values=np.array([0, 1]), 
        weights=np.array([0.4, 0.6]), 
        quantiles=0.5).item()
    
    # Ties are broken to the left
    assert 0 == nafi.weighted_quantile(
        values=np.array([0, 1]),
        weights=np.array([1, 1]),
        quantiles=0.5).item()
    
    # Extreme quantiles handled correctly
    assert 1 == nafi.weighted_quantile(
        values=np.array([0, 1]),
        weights=np.array([1, 1]),
        quantiles=1).item()
    assert 0 == nafi.weighted_quantile(
        values=np.array([0, 1]),
        weights=np.array([1, 1]),
        quantiles=0).item()



def test_weighted_ps():
    def _test(x, w, p):
        np.testing.assert_array_almost_equal(
            nafi.weighted_ps(
                np.asarray(x), 
                np.asarray(w)/np.sum(w)),
            np.asarray(p))

    # Simple test with equal weights
    _test(
        x=[0, 0, 0, 3, 4,], 
        w=[1, 1, 1, 1, 1,],
        p=[1, 1, 1, 0.4, 0.2])

    # Order of values doesn't matter
    _test(
        x=[0, 4, 0, 3, 0,], 
        w=[1, 1, 1, 1, 1,],
        p=[1, 0.2, 1, 0.4, 1])

    # Repeated values are handled correctly
    _test(
        x=[0, 4, 4, 3, 0,], 
        w=[1, 1, 1, 1, 1,],
        p=[1, 0.4, 0.4, 0.6, 1])

    # Weights don't have to be equal
    _test(
        x=[0, 4, 4, 3,], 
        w=[2, 1, 1, 1,],
        p=[1, 0.4, 0.4, 0.6])
