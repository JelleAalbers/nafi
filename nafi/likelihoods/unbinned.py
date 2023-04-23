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
import numpy as np
from scipy import stats
from tqdm import tqdm

import nafi
export, __all__ = nafi.exporter()


@partial(jax.jit, static_argnames=('n_trials', 'n_max', 'poisson'))
def drs_one_source(*, mu, key, n_trials, n_max, 
                   mu_sig_hyp, mu_bg,
                   sigma_sep,
                   x_loc=0, poisson=True,):
    """Return sum of log differential rate for events from _one_ source,
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
        dt = jax.random.exponential(key, shape=(n_trials, n_max))
        t = jnp.cumsum(dt, axis=1)
        present = t < mu
    else:
        # Keep a fixed number of events
        present = (jnp.arange(n_max) < mu)[None, :]
    
    # Differential rate (n_trials, n_max, n_hyp) 
    dr = (
        mu_sig_hyp[None,None,:] * jax.scipy.stats.norm.pdf(x, loc=0, scale=1)[...,None]
        +
        (mu_bg * jax.scipy.stats.norm.pdf(x, loc=sigma_sep, scale=1)[...,None])
    )
    
    # Return second term in log likelihood (n_trials, n_hyp)
    return jnp.sum(jax.scipy.special.xlogy(present[...,None], dr), axis=1)


@export
def get_lnl(mu_sig_hyp, mu_bg, sigma_sep,
            n_sig_max=None,
            seed=None,
            trials_per_n=10_000,
            progress=False):
    """Return (lnl, toy_weight), both (n_trials, n_hypotheses) arrays
        lnl contains the log likelihood at each hypothesis,
        toy_weight is an (n_trials, hypotheses) containing (hypothesis-dependent)
        weighting factors for each toy.

    The model is a sum of two Gaussians, one for signal and one for background,
    separated by sigma_sep. The signal rate is a hypothesis parameter.
    """
    n_hyp = len(mu_sig_hyp)
    if n_sig_max is None:
        # Highest n that can reasonably occur under the largest hypothesis
        n_sig_max = int(stats.poisson(mu_sig_hyp.max()).ppf(0.999))
    n_sig_range = np.arange(n_sig_max)

    n_sig_range = jnp.arange(n_sig_max)
    n_bg_max = int(np.ceil(stats.poisson(mu_bg).ppf(0.999)))

    # Do simulation and ln L computation with jax. It's gonna be fast...    
    if seed is None:
        seed = np.random.randint(2**32)
    key = jax.random.PRNGKey(seed=seed)

    common_kwargs = dict(
        mu_sig_hyp=mu_sig_hyp,
        mu_bg=mu_bg,
        sigma_sep=sigma_sep,
        n_trials=trials_per_n,)

    # Get lnls for background events
    # (n_trials, n_hyp)
    key, subkey = jax.random.split(key)
    lnl_bg = np.asarray(-mu_bg + drs_one_source(
        mu=mu_bg, 
        n_max=n_bg_max,
        x_loc=sigma_sep,
        poisson=True,
        key=subkey,
        **common_kwargs))

    # Get lnls for different signal event counts
    # (n_sig, n_trials, n_hyp)
    lnl_sig = np.zeros((n_sig_max, trials_per_n, len(mu_sig_hyp)))
    for n_sig in nafi.utils.tqdm_maybe(progress)(
            n_sig_range, desc='lnl simulations', leave=False):
        key, subkey = jax.random.split(key)
        lnl_sig[n_sig] = (
            lnl_bg +
            -mu_sig_hyp[None,:]
            + drs_one_source(
                mu=n_sig,
                n_max=n_sig_max,
                x_loc=0,
                poisson=False,
                key=subkey,
                **common_kwargs))

    # P(n_sig|mu_sig), (n_sig, n_mu)
    # The range of n may be insufficient for all mu, so normalize explicitly
    p_n_mu = stats.poisson(mu=mu_sig_hyp[None,:]).pmf(n_sig_range[:,None])    
    p_n_mu /= p_n_mu.sum(axis=0)[None,:]

    # Weights of individual toys
    # (n_sig, n_trials, n_mu)
    toy_weight = (p_n_mu / trials_per_n)[:,None,:].repeat(trials_per_n, axis=1)

    # Squash the (n_sig, n_trials) dimensions into one
    toy_weight = toy_weight.reshape(-1, n_hyp)
    lnl_sig = lnl_sig.reshape(-1, n_hyp)

    return lnl_sig, toy_weight