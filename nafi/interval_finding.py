from functools import partial

import nafi

import jax
import jax.numpy as jnp

export, __all__ = nafi.exporter()


@export
@partial(jax.jit, static_argnames=('interpolate',))
def intervals(
        ps, hypotheses, interpolate=True, cl=0.9):
    """Set confidence intervals on hypotheses based on p-values.

    Args:
        ps: p-values, array of shape (n_trials, n_hypotheses) or (n_hypotheses,)
        hypotheses: array of hypotheses, shape (n_hypotheses,)
        interpolate (bool): if True, use interpolation to estimate the intervals
            more precisely.
        cl (float): confidence level

    Returns:
        (ul, ll), two (n_trials,) arrays with upper and lower limits
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
        # Note we use [1,0] instead of [0,1]: the ps are decreasing w hypotheses,
        # (at ul_i, next hypothesis is not allowed by construction)
        # but jnp.interp expects increasing x.
        i = ul_i[:,None] + jnp.array([1,0])[None,:]
        ul = jnp.where(
            ul_i == len(hypotheses) - 1,
            ul,
            nafi.find_root_vec(x=hypotheses, y=ps, y0=alpha, i=i))

        # Same for lower limit
        i = ll_i[:,None] + jnp.array([-1,0,1])[None,:]
        i = jnp.clip(i, 0, len(hypotheses) - 1)
        ll = jnp.where(
            ll_i == 0,
            ll,
            nafi.find_root_vec(x=hypotheses, y=ps, y0=alpha, i=i))

    # Set empty intervals to NaN
    ul = jnp.where(empty_interval, jnp.nan, ul)
    ll = jnp.where(empty_interval, jnp.nan, ll)

    if single_trial:
        ul = ul[0]
        ll = ll[0]

    return ll, ul
