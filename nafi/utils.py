import typing

import numpy as np
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


jax_or_np_array = typing.Union[np.ndarray, jnp.ndarray]
__all__ += ['jax_or_np_array']


@export
@jax.jit
def lnl_to_weights(lnl):
    """Compute weights from log likelihoods, assuming normal outcome weighting.

    Arguments:
        lnl: log likelihoods, (n_outcome, n_hyp) array

    Returns:
        weights: (n_outcome, n_hyp) array
    """
    weights = jnp.exp(lnl)
    weights /= jnp.sum(weights, axis=0, keepdims=True)
    return weights


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
    """Return probability of getting a value equal or higher than x.

    Arguments:
        x: values, 1d array
        w: weights, 1d array
    """
    return _weighted_ps_presorted(*_order_and_index_1d(x), w)


@jax.jit
def _weighted_ps_presorted(order, sort_index, w):
    w = w[order]
    # P of getting a x lower in the sort order
    p_ordered = jnp.cumsum(w) - w
    # P of getting x equal or higher
    p_ordered = 1 - p_ordered
    return p_ordered[sort_index]


@export
@jax.jit
def order_and_index(ts):
    """Return outcome-dependent ordering and ranking of ts.

    Arguments:
      ts: test statistics, (outcomes, hypotheses) array

    Returns a tuple (order, sort_index), both of arrays of the same shape as ts.
        Here order is the number of outcomes with lower t for each hypothesis,
        and sort_index is the rank order of the outcome for each hypothesis,
        with equal outcomes assigned the same, lowest, rank.
    """
    return jax.vmap(_order_and_index_1d, in_axes=1, out_axes=1)(ts)


@jax.jit
def _order_and_index_1d(x):
    # indices that would sort x
    order = jnp.argsort(x)

    # Indices where you would place each x in sorted array
    # If all t are distinct, this would be equal to the rank order
    # (jnp.argsort(order), if I'm not mistaken)
    sort_index = jnp.searchsorted(x[order], x).clip(0, len(x))

    return order, sort_index


@export
@jax.jit
def find_root_vec(y0, y, x, i):
    """Return array of interpolated x[i] where y[:,i] == y0

    If y is (a, b), x should be (b,), and the result will be (a,).
    i is an integer array (a, c) with indices into the (b) dimension.

    This is useful with e.g. c = 2 or 3 to select a neighbourhood around the
    root.

    y[:,c] _must_ be increasing in the second dimension! This is a limitation
    we inherit from jnp.interp.
    """
    i = jnp.clip(i, 0, len(x) - 1)
    return jax.vmap(jnp.interp, in_axes=(None, 0, 0))(
        y0,                        # x
        jax.vmap(jnp.take)(y, i),  # xp
        x[i]                       # fp
    )



@export
def tqdm_maybe(progress=False):
    # The unused **kwargs is relevant, it ignores other arguments tqdm takes
    # (like total and desc)
    return tqdm if progress else lambda x, **kwargs: x


@export
def large_n_for_mu(mu):
    return int(mu + 5 * mu**0.5 + 5)


@export
def poisson_ul(n, cl=0.9):
    """Return upper limit for a Poisson process with n observed events.

    Arguments:
        n: number of observed events
        cl: confidence level
    """
    # Analytical solution for classical Poisson upper limit
    # (not the same as upper endpoint of FC interval)
    return stats.chi2.ppf(cl, 2 * n + 2) / 2
