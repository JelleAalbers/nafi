nafi
=====


![Build Status](https://github.com/JelleAalbers/nafi/actions/workflows/pytest.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nafi/badge/?version=latest)](https://nafi.readthedocs.io/en/latest/?badge=latest)

Nafi computes (non-asymptotic) frequentist confidence intervals and p-values from likelihoods.
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
# The default test statistic is t = L(hypothesis)/L(bestfit)
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

  1. Nafi is **not a modelling, fitting, simulation, or caching package**. We just turn likelihoods into p-values and confidence intervals. If you have nuisance parameters, you must take care of choosing appropriate values for the toys, then profiling (or marginalizing) them out in the toy and real data.

        Nafi does include a few example likelihoods, such as the simple (single-bin) Poisson counting experiment in the synopsis. We also have an unbinned rare-event likelihood with signal and background event positions distributed normally, a counting experiment with an uncertain background, etc.

        If you want a complete integrated framework, check out [zfit](https://github.com/zfit/zfit)/[hepstats](https://github.com/scikit-hep/hepstats), or [pyhf](https://github.com/scikit-hep/pyhf), or [RooFit](https://root.cern/manual/roofit/)/[RooStats](https://twiki.cern.ch/twiki/bin/view/RooStats/WebHome)/[HistFactory](https://twiki.cern.ch/twiki/bin/view/RooStats/HistFactory). These can all do non-asymptotic frequentist inference too.
        
  2. Nafi's **accuracy is limited by the hypothesis array you provide**. For example:

      * When `nafi.ts_and_ps` computes the maximum likelihood, it basically takes the maximum over the provided hypotheses. It does not recompute the likelihood in a minimizer loop.
      * While `nafi.intervals` interpolates between the p-values of different hypotheses, it does not recompute the likelihood to find the precise point where p = 0.1. 
      * In the synopsis example, we also get inaccurate results if the observed n approaches ~50, since we only considered mu_sig up to 42 (with mu_bg = 10).

      <br/>

      This is a an intentional choice. Frequentist constructions have to evaluate the likelihood of many toy datasets on a finely spaced grid of hypotheses to accurately capture sharp shifts in the test statistic distribution. Thus, there is little advantage in using an advanced, possibly fragile minimizer to find the best-fit on each toy.
      
      Similarly, recomputing the likelihood to zoom in on the p=0.1 intersection is not enough -- you must also know if and how the test statistic distribution changes. If you need more accuracy, compute new toys on a fine grid near your observed p=0.1 point.
