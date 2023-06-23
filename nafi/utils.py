from functools import partial

import jax
import jax.numpy as jnp
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
    return _weighted_ps_presorted(*_order_and_index(x), w)


@jax.jit
def _order_and_index(x):
    # indices that would sort x
    order = jnp.argsort(x)

    # Indices where you would place each x in sorted array
    # If all t are distinct, this is equal to the rank order
    # = jnp.argsort(order) ?
    sort_index = jnp.searchsorted(x[order], x)#.clip(0, len(x))

    return order, sort_index


@jax.jit
def _weighted_ps_presorted(order, sort_index, w):
    w = w[order]
    # P of getting a x lower in the sort order
    p_ordered = jnp.cumsum(w) - w
    # P of getting x equal or higher
    p_ordered = 1 - p_ordered
    return p_ordered[sort_index]


@export
def tqdm_maybe(progress=False):
    return tqdm if progress else lambda x, **kwargs: x


@export
def large_n_for_mu(mu):
    return int(mu + 5 * mu**0.5 + 5)


@export
def poisson_ul(n, cl=0.9):    
    """Return upper limit for a Poisson process with n observed events.

    Arguments:
     - n: number of observed events
     - cl: confidence level
    """
    # Analytical solution for classical Poisson upper limit
    # (not the same as upper endpoint of FC interval)
    return stats.chi2.ppf(cl, 2 * n + 2) / 2
