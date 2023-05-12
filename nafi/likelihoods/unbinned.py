"""Methods for producing likelihood ratios for an unbinned likelihoods
without nuisance parameters.

The model is a 

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
class UnbinnedSignalBackground:
    """Simulation and (extended) unbinned log likelihood for
    two Poisson processes (signal and background) distinguished
    by a single observable per event (e.g. energy or position).
    """

    required_params = tuple()

    def check_params(self, params):
        if not params and not self.required_params:
            return
        present = list(params.keys())
        missing = set(self.required_params) - set(present)
        if missing:
            raise ValueError(f"Missing required parameters {missing}")        

    def lnl_and_weights(
            self,
            mu_sig_hyp, 
            mu_bg,
            key=None,
            n_sig_max=None,
            n_bg_max=None,
            trials_per_n=10_000,
            **params):
        """Return (lnl, toy_weight), both (n_trials, n_hypotheses) arrays
            lnl contains the log likelihood at each hypothesis,
            toy_weight is an (n_trials, hypotheses) containing (hypothesis-dependent)
            weighting factors for each toy.
        """
        self.check_params(params)
        # If not specified, set ns to highest number that can reasonably occur
        if n_sig_max is None:
            n_sig_max = nafi.large_n_for_mu(np.max(mu_sig_hyp))
        if n_bg_max is None:
            n_bg_max = nafi.large_n_for_mu(np.max(mu_bg))
        if key is None:
            # Get a random jax key from numpy's global random state
            # Probably jax devs would not recommend this
            seed = np.random.randint(2**32)
            key = jax.random.PRNGKey(seed=seed)
        return self._lnl_and_weights(
            mu_sig_hyp=mu_sig_hyp, 
            mu_bg=mu_bg, 
            key=key, 
            n_sig_max=n_sig_max,
            n_bg_max=n_bg_max, 
            trials_per_n=trials_per_n,
            params=params)


    @partial(jax.jit, 
             static_argnames=('self', 'trials_per_n', 'n_sig_max', 'n_bg_max'))
    def _lnl_and_weights(
            self,
            mu_sig_hyp, 
            mu_bg, 
            key,
            n_sig_max,
            n_bg_max,
            trials_per_n,
            params):
        n_hyp = len(mu_sig_hyp)
        n_nsig = n_sig_max + 1
        n_sig_range = jnp.arange(n_nsig)

        common_kwargs = dict(
            n_trials=trials_per_n,
            mu_sig_hyp=mu_sig_hyp,
            mu_bg=mu_bg,
            params=params)

        # Get lnls for background events
        # (n_trials, n_hyp)
        key, subkey = jax.random.split(key)
        lnl_bg = -mu_bg + self._drs_one_source(
            mu=mu_bg, 
            key=subkey,
            n_max=n_bg_max,
            simulate=self.simulate_background,
            poisson=True,
            **common_kwargs)

        # Get lnls for different signal event counts
        # (n_sig, n_trials, n_hyp)
        lnl_sig = jnp.zeros((n_sig_max, trials_per_n, len(mu_sig_hyp)))
        # Map _drs_one_source over n_sig and RNG keys, nothing else
        key, subkey = jax.random.split(key)
        drs_alln = jax.vmap(
            partial(
                self._drs_one_source,
                n_max=n_sig_max,
                simulate=self.simulate_signal,
                poisson=False,
                **common_kwargs))
        lnl_sig = (
            lnl_bg[None,:,:] 
            - mu_sig_hyp[None,:]
            + drs_alln(
                n_sig_range,                               # mu
                jax.random.split(subkey, num=n_nsig))      # key
        )

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


    def _drs_one_source(
            self, mu, key, n_max, simulate, poisson, n_trials, mu_sig_hyp, mu_bg,
            params):
        """Return sum of log differential rate for events from _one_ source
            i.e. the second term in the unbinned ln L computation
            for the events from _one_ source only.
        
        Result is an (n_trials, n_hypotheses) array
        
        Arguments:
        - n_max: maximum n events to generate. Behaves as though extra events
            are discarded
        - poisson: if False, always make mu events instead of Poisson sampling        
        """
        # Draw event positions: (n_trials, n_max) array
        key, subkey = jax.random.split(key)
        x = simulate(n_trials, n_max, subkey, params)
        
        if poisson:
            # Draw event times, reject events beyond the dataset
            # (Would be clearer if we draw from exp with mean 1/mu, then demand < 1)
            key, subkey = jax.random.split(key)
            dt = jax.random.exponential(subkey, shape=(n_trials, n_max))
            t = jnp.cumsum(dt, axis=1)
            present = t < mu
        else:
            # Keep a fixed number of events
            # We still generated n_max events above so mu doesn't have to be
            # a static argument (and mus won't trigger recompilation)
            present = (jnp.arange(n_max) < mu)[None, :]
        
        # Differential rate (n_trials, n_max, n_hyp)
        dr = self.differential_rate(
            x[...,None], mu_sig_hyp[None,None,:], mu_bg, params)
        
        # Return second term in log likelihood (n_trials, n_hyp)
        return jnp.sum(jax.scipy.special.xlogy(present[...,None], dr), axis=1)
    
    def single_lnl(self, x, mu_sig, mu_bg, **params):
        """Return log likelihood for a single observation

        Arguments:
            x: Observed values (n_events,) array
            mu_sig: Rate hypotheses, (n_hypotheses,) array
            mu_bg: Background rate (scalar)

        Returns:
            (n_hypotheses,) array of log likelihoods
        """
        self.check_params(params)
        return self._single_lnl(x, mu_sig, mu_bg, params)

    @partial(jax.jit, static_argnames=('self',))
    def _single_lnl(self, *, x, mu_sig, mu_bg, params):
        # Get (n_events, n_hyp) array of differential rates
        dr = self.differential_rate(
            x[:,None], mu_sig[None,:], mu_bg, self.sigma_sep, params)

        return -(mu_sig + mu_bg) + jnp.sum(jnp.log(dr), axis=0)

    # Functions to override in child classes

    def simulate_signal(self, n_trials, n_max, key, params):
        """Simulate (n_trials, n_max) signal events"""
        raise NotImplementedError
    
    def simulate_background(self, n_trials, n_max, key, params):
        """Simulate (n_trials, n_max) background events"""
        raise NotImplementedError
    

@export
class TwoGaussians(UnbinnedSignalBackground):
    """Simulation and (extended) unbinned log likelihood for
        two Poisson processes with a single observable per event:
            signal: x ~ Normal(mean=0, stdev=1)
            sackground: x ~ Normal(mean=sigma_sep, stdev=1)
    """

    required_params = ('sigma_sep',)

    def simulate_signal(self, n_trials, n_max, key, params):
        return jax.random.normal(key, shape=(n_trials, n_max))
    
    def simulate_background(self, n_trials, n_max, key, params):
        return (
            self.simulate_signal(n_trials, n_max, key, params) 
            + params['sigma_sep'])

    def differential_rate(self, x, mu_sig, mu_bg, params):
        return (
            mu_sig * jax.scipy.stats.norm.pdf(x, loc=0, scale=1)
            +
            mu_bg * jax.scipy.stats.norm.pdf(x, loc=params['sigma_sep'], scale=1)
        )
