"""Classes for modelling likelihood ratios for an unbinned likelihoods
without nuisance parameters.
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

    To specify your own signal and background distributions, inherit from this
    class and override the simulate_signal, simulate_background, and
    differential_rate methods. See the TwoGaussians class for an example.
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
            return_outcomes=False,
            **params):
        """Return (logl, toy weight) for an unbinned signal/background likelihood

        Both are (n_outcomes, hypotheses) arrays:
                lnl contains the log likelihood at each hypothesis,
                toy_weight contains P(outcome | hypotheses), normalized over outcomes

        Arguments:
            mu_sig_hyp: Array with expected signal event hypotheses
            mu_bg: Expected background events (scalar)
            key: Jax PNRG key to use. If not provided, will choose a random one
                according to the numpy global random state.
            n_sig_max: Largest number of signal events to consider.
                If None, will be determined automatically from mu_sig.
            n_bg_max: Largest number of background events to consider.
                If None, will be determined automatically from mu_bg.
            trials_per_n: Number of MC trials to use per signal event count.
            return_outcomes: If True, return a third array of shape (n_outcomes,)
                containing a summary statistic of the outcome for each toy.
                The default (set by the summary_stat method) is the total number
                of observed events.
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
        lnl, weights, summary = self._lnl_and_weights(
            mu_sig_hyp=mu_sig_hyp,
            mu_bg=mu_bg,
            key=key,
            n_sig_max=n_sig_max,
            n_bg_max=n_bg_max,
            trials_per_n=trials_per_n,
            params=params)
        if return_outcomes:
            return lnl, weights, summary
        return lnl, weights


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
        # drs_bg is (n_trials, n_hyp), summary_bg is (n_trials,)
        key, subkey = jax.random.split(key)
        drs_bg, summary_bg = self._drs_one_source(
            mu=mu_bg,
            key=subkey,
            n_max=n_bg_max,
            simulate=self.simulate_background,
            poisson=True,
            **common_kwargs)
        lnl_bg = -mu_bg + drs_bg

        # Get lnls for different signal event counts
        # Map _drs_one_source over n_sig and RNG keys, nothing else
        key, subkey = jax.random.split(key)
        drs_alln = jax.vmap(
            partial(
                self._drs_one_source,
                n_max=n_sig_max,
                simulate=self.simulate_signal,
                poisson=False,
                **common_kwargs),
            out_axes=(0, 0))
        # drs_sig is (n_sig, n_trials, n_hyp), same as lnl
        # summary_sig is (n_sig, n_trials)
        drs_sig, summary_sig = drs_alln(
            mu=n_sig_range,
            key=jax.random.split(subkey, num=n_nsig)
        )
        lnl = lnl_bg[None,:,:] - mu_sig_hyp[None,:] + drs_sig
        summary = summary_bg[None,:] + summary_sig

        # P(n_sig|mu_sig), (n_sig, n_mu)
        # The range of n may be insufficient for all mu, so normalize explicitly
        p_n_mu = jax.scipy.stats.poisson.pmf(k=n_sig_range[:,None], mu=mu_sig_hyp[None,:])
        p_n_mu /= p_n_mu.sum(axis=0)[None,:]

        # Weights of individual toys
        # (n_sig, n_trials, n_mu)
        toy_weight = (p_n_mu / trials_per_n)[:,None,:].repeat(trials_per_n, axis=1)

        # Squash the (n_sig, n_trials) dimensions into one
        toy_weight = toy_weight.reshape(-1, n_hyp)
        lnl = lnl.reshape(-1, n_hyp)
        summary = summary.reshape(-1)

        return lnl, toy_weight, summary

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
            # We still generated n_max events above since
            # jax won't let us  specialize shapes based on argument values
            present = (jnp.arange(n_max) < mu)[None, :]

        # Differential rate (n_trials, n_max+1, n_hyp)
        dr = self.differential_rate(
            x[...,None], mu_sig_hyp[None,None,:], mu_bg, params)

        # Second term in log likelihood (n_trials, n_hyp)
        logdr_sum = jnp.sum(jax.scipy.special.xlogy(present[...,None], dr), axis=1)

        # Summary statistic (n_trials,)
        summary = self.summary(x, present, params)
        return logdr_sum, summary

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

    def summary(self, x, present, params):
        """Return a summary statistic of the data.

        The number must be additive over events. This function is called
        once for signal events and once for background events, then the results
        are added.

        By default, returns the total number of events.
        """
        # x and present are both (n_trials, n_ns) arrays
        return jnp.sum(present, axis=1)

    def simulate_signal(self, n_trials, n_max, key, params):
        """Simulate (n_trials, n_max) signal events"""
        raise NotImplementedError

    def simulate_background(self, n_trials, n_max, key, params):
        """Simulate (n_trials, n_max) background events"""
        raise NotImplementedError

    def differential_rate(self, x, mu_sig, mu_bg, params):
        """Return differential rate for events with observed x
        """
        raise NotImplementedError


@export
class TwoGaussians(UnbinnedSignalBackground):
    """Simulation and (extended) unbinned log likelihood for a signal and
    background that are both Gaussians with unit variance, but different means.

    Takes a single parameter, ``sigma_sep``, the distance between the signa.
    and background means.
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
