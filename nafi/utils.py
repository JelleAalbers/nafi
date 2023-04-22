import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from tqdm import tqdm


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def find_root_vec(y, x=None, guess_i=None, y0=None):
    """Estimate location where y = y0. Returns an y.shape[:-1] array.
    
    If x is provided, it should match the last dimension of x.
        The result will be the x-value at which y = y0.
    Otherwise, return index into an x-array you provide later.

    Arguments:
     - y: array of values
     - x: if specified, instead of returning interpolated index,
          interpolate return value linearly in x.
     - guess_i: index at which to start searching.
           If not provided, will use argmin(abs(y), axis=-1)
     - y0: will be subtracted from y before finding root
    """
    if y0 is not None:
        y = y - y0
    if guess_i is None:
        guess_i = np.argmin(np.abs(y), axis=-1)
    assert guess_i.shape == y.shape[:-1]

    largest_i = y.shape[-1] - 1
    before_i = (guess_i - 1).clip(0, largest_i)
    after_i = (guess_i + 1).clip(0, largest_i)

    before_val, guess_val, after_val = [
        np.take_along_axis(y, indices=idx[...,None], axis=-1)[...,0]
        for idx in (before_i, guess_i, after_i)]

    # TODO: fails for a few lower limits.. 
    # assert np.all(
    #     (guess_i == 0) 
    #     | (guess_i == largest_i) 
    #     | (np.sign(before_val) != np.sign(after_val)))
    
    root_is_left = np.sign(after_val) == np.sign(guess_val)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        itp_i = np.where(
            root_is_left, 
            # before_i + (0 - before_val)/(max_val - before_val) * (max_i - before_i),
            before_i - before_val/(guess_val - before_val),
            #max_i + (0 - max_val)/(after_val - max_val) * (after_i - max_i),
            guess_i - guess_val/(after_val - guess_val),
        )
    itp_i = np.where(root_is_left & (guess_i == 0), 0, itp_i)
    itp_i = np.where((~root_is_left) & (guess_i == largest_i), largest_i, itp_i)
    
    # In cases where there actually is no sign change, the above gives weird results
    # Revert to the guess in this case.
    no_change = np.sign(after_val) == np.sign(guess_val)
    itp_i = np.where(no_change, guess_i, itp_i)
    
    if x is not None:
        if x.shape == y.shape[-1:]:
            return np.interp(x=itp_i, xp=np.arange(len(x)), fp=x)
        # Works, I think
        # elif x.shape == y.shape:
        #     x_before, x_guess, x_after = [
        #         np.take_along_axis(x, indices=idx[...,None], axis=-1)[...,0]
        #         for idx in (before_i, guess_i, after_i)]
        #     return np.where(
        #         root_is_left,
        #         x_before + (x_guess - x_before) * (itp_i - before_i),
        #         x_guess + (x_after - x_guess) * (itp_i - guess_i))
        raise ValueError("x and y must have matching final axis length")
    return itp_i

### Unit tests for find_root_vec
# Test 1: simple case
np.testing.assert_array_almost_equal(
    find_root_vec(
        x=np.arange(5), 
        y=np.arange(5)[None,:] - 6.5 + np.arange(9)[:,None]),
    np.array([4, 4, 4, 3.5, 2.5, 1.5, 0.5, 0, 0]))


# Adapted from https://stackoverflow.com/a/29677616
@export
@jax.jit
def weighted_quantile(values, weights, quantiles, values_sorted=False):
    """Compute quantiles for weighted values
    :param values: numpy.array with data
    :param weights: array-like of the same length as `values`
    :param quantiles: array-like with many quantiles needed.
        Should all be in [0,1].    
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :return: array with computed quantiles.
    """
    if not values_sorted:
        sorter = jnp.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    # Original code had - 0.5 * weights here, which fails the test below
    weighted_quantiles = jnp.cumsum(weights)
    weighted_quantiles /= jnp.sum(weights)
    return jnp.interp(quantiles, weighted_quantiles, values)

# Test quantiles are not interpolated
assert 0. == weighted_quantile(
    values=np.array([0, 1]), 
    weights=np.array([0.6, 0.4]), 
    quantiles=0.5).item()


# Sufficiently slow that jax.jit is worth it
@export
@jax.jit
def weighted_ps(x, w):
    # indices that would sort x
    order = jnp.argsort(x)
    
    # P of getting a x lower in the sort order
    p_ordered = jnp.cumsum(w[order]) - w[order]
    
    # Indices where you would place each t in sorted array
    # If all t are distinct, this is equal to the rank_order
    sort_index = jnp.searchsorted(x[order], x)#.clip(0, len(x))
    
    return p_ordered[sort_index]

np.testing.assert_array_almost_equal(
    weighted_ps(
        x=np.array([0, 0, 0, 3, 4,]), 
        w=np.array([1, 1, 1, 1, 1,])/5),
    np.array([0, 0, 0, 0.6, 0.8]))

np.testing.assert_array_almost_equal(
    weighted_ps(
        x=np.array([0, 4, 0, 3, 0,]), 
        w=np.array([1, 1, 1, 1, 1,])/5),
    np.array([0, 0.8, 0, 0.6, 0]))

np.testing.assert_array_almost_equal(
    weighted_ps(
        x=np.array([0, 4, 4, 3, 0,]), 
        w=np.array([1, 1, 1, 1, 1,])/5),
    np.array([0, 0.6, 0.6, 0.4, 0]))

np.testing.assert_array_almost_equal(
    weighted_ps(
        x=np.array([0, 4, 4, 3,]), 
        w=np.array([2, 1, 1, 1,])/5),
    np.array([0, 0.6, 0.6, 0.4]))


@export
def tqdm_maybe(progress=False):
    return tqdm if progress else lambda x, **kwargs: x


@export
def toy_weights(*, shape, p_n_mu=None, hypotheses=None):
    trials_per_n, n_sig_max, *_ = shape
    n_sig_range = np.arange(n_sig_max)
    if p_n_mu is None:
        if hypotheses is None:
            raise ValueError("Provide either p_n_mu or hypotheses")
        # Assume hypotheses are signal rates
        p_n_mu = stats.poisson(mu=hypotheses[None,:]).pmf(n_sig_range[:,None])

    # The range of n may be insufficient, so need to upweight toys to compensate
    p_n_mu /= p_n_mu.sum(axis=0)[None,:]

    toy_weight = p_n_mu / trials_per_n
    return toy_weight[None,...]


# Unit test: toy_weights sums to 1 for each hypothesis (even if p_n_mu does not)

