import numpy as np

import nafi


def test_find_root_vec():
    np.testing.assert_array_almost_equal(
        nafi.find_root_vec(
            x=np.arange(5), 
            y=np.arange(5)[None,:] - 6.5 + np.arange(9)[:,None]),
        np.array([4, 4, 4, 3.5, 2.5, 1.5, 0.5, 0, 0]))


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


def test_weighted_ps():
    # Simple test with equal weights
    np.testing.assert_array_almost_equal(
        nafi.weighted_ps(
            x=np.array([0, 0, 0, 3, 4,]), 
            w=np.array([1, 1, 1, 1, 1,])/5),
        np.array([0, 0, 0, 0.6, 0.8]))

    # Order of values doesn't matter
    np.testing.assert_array_almost_equal(
        nafi.weighted_ps(
            x=np.array([0, 4, 0, 3, 0,]), 
            w=np.array([1, 1, 1, 1, 1,])/5),
        np.array([0, 0.8, 0, 0.6, 0]))

    # Repeated values are handled correctly
    np.testing.assert_array_almost_equal(
        nafi.weighted_ps(
            x=np.array([0, 4, 4, 3, 0,]), 
            w=np.array([1, 1, 1, 1, 1,])/5),
        np.array([0, 0.6, 0.6, 0.4, 0]))

    # Weights don't have to be equal
    np.testing.assert_array_almost_equal(
        nafi.weighted_ps(
            x=np.array([0, 4, 4, 3,]), 
            w=np.array([2, 1, 1, 1,])/5),
        np.array([0, 0.6, 0.6, 0.4]))
