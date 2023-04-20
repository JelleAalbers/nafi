import numpy as np
import nafi
import warnings

export, __all__ = nafi.exporter()

@export
def intervals(
        ps, hypotheses, interpolate=True, cl=0.9):
    """Convert p-values to confidence intervals

    Args:
        ps (np.ndarray): p-values, shape (n_trials, n_hypotheses)
        hypotheses (np.ndarray): array of hypotheses, shape (n_hypotheses,)
        interpolate (bool): if True, use interpolation to estimate the intervals
            more precisely.
        cl (float): confidence level

    Returns:
        (ul, ll), each arrays with shape of ps without the last axis.
    """
    # is hyp allowed by the trial?  (|trials|,|mu|)
    alpha = 1 - cl
    allowed = ps >= alpha

    empty_interval = np.sum(allowed, axis=-1) == 0
    if np.any(empty_interval):
        warnings.warn(
            f"Warning: {np.sum(empty_interval)} trials have empty intervals; "
            "upper and lower limits are returned as NaN.")

    # Limits = lowest & highest allowed hypothesis
    # flat (n_trials,) arrays
    ul_i = np.argmax(
        np.asarray(hypotheses)[None,:] * allowed,
        axis=-1)
    ll_i = np.argmin(
        np.where(
            allowed,
            np.asarray(hypotheses)[None,:],
            np.inf),
        axis=-1)
    
    # Temporarily set empty intervals to 0, so indexing works
    # ul_i = np.where(empty_interval, 0, ul_i)
    # ll_i = np.where(empty_interval, 0, ll_i)

    # Interpolate / fine-tune limits
    if not interpolate:
        ul = hypotheses[ul_i]
        ll = hypotheses[ll_i]

    else:
        # TODO: if we add +1 offset, get a bit of overcoverage. Why?
        ul = nafi.utils.find_root_vec(y=ps, x=hypotheses, guess_i=ul_i, y0=alpha)
        # TODO: there are some bugs here...
        # TODO: the -1 offset seems necessary to guarantee ll <= ll_coarse. But why?
        highest_i = len(hypotheses) - 1
        ll = nafi.utils.find_root_vec(y=ps, x=hypotheses, guess_i=(ll_i - 1).clip(0, highest_i), y0=alpha)
        # TODO: the interpolation sometimes invents discoveries even when 0 is allowed,
        # this is a rough fix:
        ll = np.where(allowed[...,0], 0, ll)

    # Set empty intervals to NaN. Choose another method please...
    ul = np.where(empty_interval, np.nan, ul)
    ll = np.where(empty_interval, np.nan, ll)

    return ul, ll
