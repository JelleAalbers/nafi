nafi
=====


![Build Status](https://github.com/JelleAalbers/nafi/actions/workflows/pytest.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nafi/badge/?version=latest)](https://nafi.readthedocs.io/en/latest/?badge=latest)

Nafi computes (non-asymptotic) frequentist p-values and confidence intervals from likelihoods.
CLs and simple Bayesian methods are also supported.

[JAX](https://github.com/google/jax) is used to accelerate computations, so nafi will be much faster if you have a GPU.

Synopsis
============

```python
import nafi
import numpy as np

# Hypotheses to test -- here a number of expected signal events.
# (could be WIMP cross-section, Higgs mass, ...)
hypotheses = np.arange(0, 42, 0.01)

# Get log likelihoods and weights of different (toy) outcomes.
# This is a counting experiment with signal (mu_sig) and background (mu_bg).
# >> Replace this with your likelihood / simulation code! <<
from nafi.likelihoods import counting
mu_bg = 10
lnl, weights = counting.lnl_and_weights(mu_sig=hypotheses, mu_bg=mu_bg)

# Compute test statistics and p-values
# The default test statistic is t = -2 ln[ L(hypothesis)/L(bestfit) ]
ts, ps = nafi.ts_and_pvals(lnl, weights)

# Confidence intervals on your toy data
# shape: (n_toys)
lower_limits, upper_limits = nafi.intervals(ps, hypotheses, cl=0.9)

# Compute "Brazil band", the median, ±1σ and ±2σ upper limit
brazil = nafi.brazil_band(upper_limits, weights)

# Compute results on your real data
lnl_obs = counting.single_lnl(n=17, mu_sig=hypotheses, mu_bg=mu_bg)
t_obs, p_obs = nafi.single_ts_and_pvals(lnl_obs, ts=ts, ps=ps)
ll_obs, ul_obs = nafi.intervals(p_obs, hypotheses, cl=0.9)

print(f"Discovery p-value is {p_obs[0]:.4f}")
print(f"90% confidence interval: [{ll_obs:.2f}, {ul_obs:.2f}]")
print(f"Median upper limit is {brazil[0][0]:.2f}, "
      f"68% band [{brazil[-1][0]:.2f}, {brazil[1][0]:.2f}]")
```

Gives:

```
Discovery p-value is 0.0270
90% confidence interval: [1.81, 15.02]
Median upper limit is 6.51, 68% band [3.06, 10.05]
```


Limitations
============

  1. Nafi is **not a modelling, fitting, simulation, or caching package**. We just turn likelihoods into p-values and confidence intervals. 

      In particular, if you have nuisance parameters, _you_ must take care of them: choose appropriate values for toy outcomes, profile (or marginalize) over the nuisance parameters, then pass the profile likelihood to nafi.

        
  2. Nafi's **accuracy is limited by the hypothesis array you provide**. For example:

      * When `nafi.ts_and_ps` computes the maximum likelihood, it just finds the maximum over the provided hypotheses. It does not recompute the likelihood in a minimizer loop.
      * While `nafi.intervals` does interpolate between p-values of different hypotheses, it does not recompute the likelihood to find the precise point where p = 0.1. 
      * In the synopsis example, we also get inaccurate results if the observed n approaches ~50, since we only considered mu_sig up to 42 (with mu_bg = 10).

If you want a more complete and integrated inference framework, you might like [zfit](https://github.com/zfit/zfit)/[hepstats](https://github.com/scikit-hep/hepstats), or [pyhf](https://github.com/scikit-hep/pyhf), or [RooFit](https://root.cern/manual/roofit/)/[RooStats](https://twiki.cern.ch/twiki/bin/view/RooStats/WebHome)/[HistFactory](https://twiki.cern.ch/twiki/bin/view/RooStats/HistFactory).


Examples
=========
Nafi does include some simple likelihoods you can use for testing, and if your application is sufficiently simple.
* `nafi.likelihood.counting`, the (single-bin) Poisson counting experiment shown in the synopsis;
* `nafi.likelihoods.unbinned`, for experiments where the signal and background events have different (but certain) distributions. A simple example is `nafi.likelihoods.two_gaussians`: two Gaussians of the same width, separated by a parameter `sigma_sep` times that width.
*  `nafi.likelihoods.counting_uncbg`, a counting experiment where the background expectation has an (external/theoretical) uncertainty.
