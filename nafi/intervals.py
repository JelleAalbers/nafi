from functools import partial

import nafi

import jax
import jax.numpy as jnp

export, __all__ = nafi.exporter()


@export
@partial(jax.jit, static_argnames=('interpolate',))
def intervals(
        ps, hypotheses, interpolate=True, cl=0.9):
    """Convert p-values to confidence intervals

    Args:
        ps: p-values, array of shape (n_trials, n_hypotheses) or (n_hypotheses,)
        hypotheses: array of hypotheses, shape (n_hypotheses,)
        interpolate (bool): if True, use interpolation to estimate the intervals
            more precisely.
        cl (float): confidence level

    Returns:
        (ul, ll), each arrays with shape of ps without the last axis.
    """
    single_trial = len(ps.shape) == 1
    if single_trial:
        ps = ps[None,:]

    # Do not allow hypotheses with p-values of NaN
    # (without this, the interpolation gets into trouble if p-values are e.g.
    #  NaN, 1, NaN)
    ps = jnp.nan_to_num(ps)

    # is hyp allowed by the trial?  (|trials|,|mu|)
    alpha = 1 - cl
    allowed = ps >= alpha

    empty_interval = jnp.sum(allowed, axis=-1) == 0

    # Limits = lowest & highest allowed hypothesis
    # flat (n_trials,) arrays
    ul_i = jnp.argmax(
        jnp.asarray(hypotheses)[None,:] * allowed,
        axis=-1)
    ll_i = jnp.argmin(
        jnp.where(
            allowed,
            jnp.asarray(hypotheses)[None,:],
            jnp.inf),
        axis=-1)
    
    ul = hypotheses[ul_i]
    ll = hypotheses[ll_i]

    # Interpolate / fine-tune limits
    if interpolate:
        # i = indices of size-2 slice of values to interpolate.
        # Note the [::-1]: ps is decreasing in hypothesis,
        # (at ul_i, next hypothesis is not allowed by construction)
        # but jnp.interp expects increasing x.
        # TODO: ... but my unit tests still pass without the [::-1]...
        i = ul_i[:,None] + jnp.arange(2)[::-1][None,:]
        ul = jnp.where(
            ul_i == len(hypotheses) - 1,
            ul,
            _itp(alpha, ps, hypotheses, i))

        # Same for lower limit
        i = ll_i[:,None] + jnp.arange(-1, 1)[None,:]
        i = jnp.clip(i, 0, len(hypotheses) - 1)
        ll = jnp.where(
            ll_i == 0,
            ll,
            _itp(alpha, ps, hypotheses, i))

    # Set empty intervals to NaN
    ul = jnp.where(empty_interval, jnp.nan, ul)
    ll = jnp.where(empty_interval, jnp.nan, ll)

    if single_trial:
        ul = ul[0]
        ll = ll[0]

    return ll, ul


def _itp(alpha, ps, hypotheses, i):
    """Return interpolated hypothesis[i] where ps[:,i] == alpha"""
    i = jnp.clip(i, 0, len(hypotheses) - 1)
    return jax.vmap(jnp.interp, in_axes=(None, 0, 0))(
        alpha,                        # x
        jax.vmap(jnp.take)(ps, i),    # xp
        hypotheses[i],                # fp
    )
