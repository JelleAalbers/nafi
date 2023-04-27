"""Methods for producing likelihood ratios for an unbinned likelihoods
without nuisance parameters.

The model is a sum of two Gaussians, one for signal and one for background.

End goal is an (BATCH, n_sig, hypothesis_i) tensor of likelihood ratios,
where 
 - BATCH is a batch dimension, i.e. over different MC toys
 - n_sig is the number of generated signal events
 - hypothesis_i indexes the hypothesis (signal rate)
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

import nafi
export, __all__ = nafi.exporter()


@export
def lnl_and_weights(
        mu_sig_hyp, 
        mu_bg, 
        sigma_sep,
        key=None,
        n_sig_max=None,
        n_bg_max=None,
        trials_per_n=10_000):
    """Return (lnl, toy_weight), both (n_trials, n_hypotheses) arrays
        lnl contains the log likelihood at each hypothesis,
        toy_weight is an (n_trials, hypotheses) containing (hypothesis-dependent)
        weighting factors for each toy.

    The model is a sum of two Gaussians, one for signal and one for background,
    separated by sigma_sep. The signal rate is a hypothesis parameter.
    """
    # Set n to highest number that can reasonably occur
    if n_sig_max is None:
        n_sig_max = nafi.large_n_for_mu(np.max(mu_sig_hyp))
    if n_bg_max is None:
        n_bg_max = nafi.large_n_for_mu(np.max(mu_bg))
    if key is None:
        # Get a random jax key from numpy's global random state
        # Probably jax devs would not recommend this
        seed = np.random.randint(2**32)
        key = jax.random.PRNGKey(seed=seed)
    return _lnl_and_weights(
        mu_sig_hyp=mu_sig_hyp, 
        mu_bg=mu_bg, 
        sigma_sep=sigma_sep, 
        key=key, 
        n_sig_max=n_sig_max,
        n_bg_max=n_bg_max, 
        trials_per_n=trials_per_n)


@partial(jax.jit, static_argnames=('trials_per_n', 'n_sig_max', 'n_bg_max'))
def _lnl_and_weights(
        mu_sig_hyp, 
        mu_bg, 
        sigma_sep,
        key,
        n_sig_max,
        n_bg_max,
        trials_per_n=10_000):
    n_hyp = len(mu_sig_hyp)
    n_nsig = n_sig_max + 1
    n_sig_range = jnp.arange(n_nsig)

    common_kwargs = dict(
        n_trials=trials_per_n,
        mu_sig_hyp=mu_sig_hyp,
        mu_bg=mu_bg,
        sigma_sep=sigma_sep)

    # Get lnls for background events
    # (n_trials, n_hyp)
    key, subkey = jax.random.split(key)
    lnl_bg = -mu_bg + _drs_one_source(
        mu=mu_bg, 
        n_max=n_bg_max,
        x_loc=sigma_sep,
        poisson=True,
        key=subkey,
        **common_kwargs)

    # Get lnls for different signal event counts
    # (n_sig, n_trials, n_hyp)
    lnl_sig = jnp.zeros((n_sig_max, trials_per_n, len(mu_sig_hyp)))
    # + 1 since key also counts
    drs_alln = jax.vmap(
        _drs_one_source, 
        # Map over n_sig and RNG keys, nothing else
        in_axes=(0,0,None,None,None,None,None,None,None))
    key, subkey = jax.random.split(key)
    # At least in my jax version, vmap does not work well with keyword arguments
    # yet: https://github.com/google/jax/issues/7465
    lnl_sig = (
        lnl_bg[None,:,:] 
        - mu_sig_hyp[None,:]
        + drs_alln(
            n_sig_range,  # mu
            jax.random.split(subkey, num=n_nsig),      # key
            n_sig_max,    # n_max
            0,            # x_loc
            False,        # poisson = False
            *common_kwargs.values()))

    # P(n_sig|mu_sig), (n_sig, n_mu)
    # The range of n may be insufficient for all mu, so normalize explicitly
    p_n_mu = jax.scipy.stats.poisson.pmf(k=n_sig_range[:,None], mu=mu_sig_hyp[None,:])
    p_n_mu /= p_n_mu.sum(axis=0)[None,:]

    # Weights of individual toys
    # (n_sig, n_trials, n_mu)
    toy_weight = (p_n_mu / trials_per_n)[:,None,:].repeat(trials_per_n, axis=1)

    # Squash the (n_sig, n_trials) dimensions into one
    toy_weight = toy_weight.reshape(-1, n_hyp)
    lnl_sig = lnl_sig.reshape(-1, n_hyp)

    return lnl_sig, toy_weight


@partial(jax.jit, static_argnames=('n_trials', 'n_max', 'poisson'))
def _drs_one_source(
        mu, key, n_max, x_loc, poisson, n_trials,
        mu_sig_hyp, mu_bg,
        sigma_sep):
    """Return sum of log differential rate for events from _one_ source
        i.e. the second term in the unbinned ln L computation
        for the events from _one_ source only.
    
    Result is an (n_trials, n_hypotheses) array
    
    Arguments:
      - sigma_sep: separation between signal and background
      - n_max: maximum n events to generate. Behaves as though extra events
          are discarded
      - x_loc: mean location of events. 0 for signal, sigma_sep for bg
      - poisson: if False, always make mu events instead of Poisson sampling        
    """
    # Draw event positions: (n_trials, n_max) array
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(n_trials, n_max)) + x_loc
    
    if poisson:
        # Draw event times, reject events beyond the dataset
        # (Would be clearer if we draw from exp with mean 1/mu, then demand < 1)
        key, subkey = jax.random.split(key)
        dt = jax.random.exponential(subkey, shape=(n_trials, n_max))
        t = jnp.cumsum(dt, axis=1)
        present = t < mu
    else:
        # Keep a fixed number of events
        present = (jnp.arange(n_max) < mu)[None, :]
    
    # Differential rate (n_trials, n_max, n_hyp)
    dr = differential_rate(x[...,None], mu_sig_hyp[None,None,:], mu_bg, sigma_sep)
    
    # Return second term in log likelihood (n_trials, n_hyp)
    return jnp.sum(jax.scipy.special.xlogy(present[...,None], dr), axis=1)


@export
@jax.jit
def differential_rate(x, mu_sig, mu_bg, sigma_sep):
    return (
        mu_sig * jax.scipy.stats.norm.pdf(x, loc=0, scale=1)
        +
        mu_bg * jax.scipy.stats.norm.pdf(x, loc=sigma_sep, scale=1)
    )


@export
@jax.jit
def single_lnl(*, x, mu_sig, mu_bg, sigma_sep):
    """Return log likelihood for a single observation

    Arguments:
        x: Observed values (n_events,) array
        mu_sig: Rate hypotheses, (n_hypotheses,) array
        mu_bg: Background rate (scalar)

    Returns:
        (n_hypotheses,) array of log likelihoods
    """
    # Get (n_events, n_hyp) array of differential rates
    dr = differential_rate(x[:,None], mu_sig[None,:], mu_bg, sigma_sep)

    return -(mu_sig + mu_bg) + jnp.sum(jnp.log(dr), axis=0)
