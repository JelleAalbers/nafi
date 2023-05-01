from functools import partial
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
def find_root_vec(y, x, guess_i):
    """Estimate location where y(x) = 0. Returns an y.shape[:-1] array.

    Arguments:
     - y: array of values
     - x: if specified, instead of returning interpolated index,
          interpolate return value linearly in x.
     - guess_i: index around which to search for root
        Will look at one value before and one value after.
    """
    di = jnp.arange(-1, 2)

    # Search only in 3-value slice around guess_i
    # (I'm sure there is a numpy indexing trick to do this without vmap...)
    # Shape (n_x, 3)
    i = guess_i[:,None] + di[None,:]
    illegal = (i < 0) | (i >= len(x))
    i = jnp.clip(i, 0, len(x)-1)
    y_slice = jax.vmap(jnp.take)(y, i)
    x_slice = x[i]

    # For illegal indices, set both x and y value to NaN
    # (especially NaNing x is important; otherwise interp gets a multi-valued
    #   function, in which case its behaviour is weird)
    y_slice = jnp.where(illegal, jnp.nan, y_slice)
    x_slice = jnp.where(illegal, jnp.nan, x_slice)

    # jnp.interp assumes x-values are sorted. :-(
    # So we need to sort them first.
    sort_index = jnp.argsort(x_slice, axis=-1)
    y_slice = jnp.take_along_axis(y_slice, sort_index, axis=-1)
    x_slice = jnp.take_along_axis(x_slice, sort_index, axis=-1)

    # jax.debug.print("i={i}, xslice={x_slice}, yslice={yslice}", i=i[32], x_slice=x_slice[32], yslice=y_slice[32])
    # Find x-value where y(x) = 0 
    return jax.vmap(jnp.interp, in_axes=(None, 0, 0))(
        0,                # x
        y_slice,          # xp
        x_slice,          # fp
        )


@export
@jax.jit
def weighted_quantile(values, weights, quantiles):
    """Compute quantiles for weighted values.

    Does not interpolate values: instead returns value whose probability mass
    contains the quantile.
    
    :param values: numpy.array with data
    :param weights: array-like of the same length as `values`
    :param quantiles: array-like with many quantiles needed.
        Should all be in [0,1].
    :return: array with computed quantiles.
    """
    order = jnp.argsort(values)
    values = values[order]
    weights = weights[order]
    return weighted_quantile_sorted(values, weights, quantiles)


@export
@jax.jit
def weighted_quantile_sorted(values, weights, quantiles):
    cdf = jnp.cumsum(weights)
    cdf /= cdf[-1]
    idx = jnp.searchsorted(cdf, quantiles)
    return values[idx]


@export
@jax.jit
def weighted_ps(x, w):
    # indices that would sort x
    order = jnp.argsort(x)
    
    # P of getting a x lower in the sort order
    w_ordered = w[order]
    p_ordered = jnp.cumsum(w_ordered) - w_ordered
    
    # Indices where you would place each x in sorted array
    # If all t are distinct, this is equal to the rank order
    # = jnp.argsort(order) ?
    sort_index = jnp.searchsorted(x[order], x)#.clip(0, len(x))
    
    return p_ordered[sort_index]


@export
def tqdm_maybe(progress=False):
    return tqdm if progress else lambda x, **kwargs: x


@export
def large_n_for_mu(mu):
    return int(mu + 5 * mu**0.5 + 5)
