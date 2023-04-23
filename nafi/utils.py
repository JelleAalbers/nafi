import warnings

import jax
import jax.numpy as jnp
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
@jax.jit
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
        guess_i = jnp.argmin(jnp.abs(y), axis=-1)
    assert guess_i.shape == y.shape[:-1]

    largest_i = y.shape[-1] - 1
    before_i = (guess_i - 1).clip(0, largest_i)
    after_i = (guess_i + 1).clip(0, largest_i)

    before_val, guess_val, after_val = [
        jnp.take_along_axis(y, indices=idx[...,None], axis=-1)[...,0]
        for idx in (before_i, guess_i, after_i)]

    # TODO: fails for a few lower limits.. 
    # assert np.all(
    #     (guess_i == 0) 
    #     | (guess_i == largest_i) 
    #     | (np.sign(before_val) != np.sign(after_val)))
    
    root_is_left = jnp.sign(after_val) == jnp.sign(guess_val)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        itp_i = jnp.where(
            root_is_left, 
            # before_i + (0 - before_val)/(max_val - before_val) * (max_i - before_i),
            before_i - before_val/(guess_val - before_val),
            #max_i + (0 - max_val)/(after_val - max_val) * (after_i - max_i),
            guess_i - guess_val/(after_val - guess_val),
        )
    itp_i = jnp.where(root_is_left & (guess_i == 0), 0, itp_i)
    itp_i = jnp.where((~root_is_left) & (guess_i == largest_i), largest_i, itp_i)
    
    # In cases where there actually is no sign change, the above gives weird results
    # Revert to the guess in this case.
    no_change = jnp.sign(after_val) == jnp.sign(guess_val)
    itp_i = jnp.where(no_change, guess_i, itp_i)
    
    if x is not None:
        if x.shape == y.shape[-1:]:
            return jnp.interp(x=itp_i, xp=jnp.arange(len(x)), fp=x)
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


@export
@jax.jit
def weighted_quantile(values, weights, quantiles, values_sorted=False):
    """Compute quantiles for weighted values.

    Does not interpolate values: instead returns value whose probability mass
    contains the quantile.
    
    :param values: numpy.array with data
    :param weights: array-like of the same length as `values`
    :param quantiles: array-like with many quantiles needed.
        Should all be in [0,1].
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :return: array with computed quantiles.
    """
    if not values_sorted:
        order = jnp.argsort(values)
        values = values[order]
        weights = weights[order]

    cdf = jnp.cumsum(weights)
    cdf /= cdf[-1]

    idx = jnp.searchsorted(cdf, quantiles)
    return values[idx]


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


@export
def tqdm_maybe(progress=False):
    return tqdm if progress else lambda x, **kwargs: x
