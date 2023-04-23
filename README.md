nafi
=====


![Build Status](https://github.com/JelleAalbers/nafi/actions/workflows/pytest.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nafi/badge/?version=latest)](https://nafi.readthedocs.io/en/latest/?badge=latest)

Nafi computes (non-asymptotic) frequentist confidence intervals and p-values from likelihoods.
CLs and simple Bayesian methods are also supported.

Synopsis
============

```python
import nafi
import numpy as np

# Hypotheses to test, e.g. number of expected signal events.
# (or WIMP cross-section, Higgs mass, ...)
mu_sig = np.linspace(0, 42, 200)

# Get log likelihoods and weights of different (toy) outcomes
# Insert your likelihood / simulation code here!
from nafi.examples import counting
mu_bg = 10
lnl, weights = counting.lnl_and_weights(mu_sig, mu_bg=10)

# Compute test statistics and p-values
# The default test statistic is t = L(hypothesis)/L(bestfit)
ts, ps = nafi.ts_and_pvals(lnl, weights)

# Confidence intervals on your toy data
# shape: (n_toys)
lower_limits, upper_limits = nafi.intervals(ps, hypotheses, cl=0.9)

# Compute "Brazil band", the median, ±1σ and ±2σ upper limit
brazil = nafi.brazil_band(upper_limits, weights)

# Compute results on your real data
# (We have alternatives that don't need the full weights/ts/ps)
lnl_obs = counting.single_lnl(n=21, mu_sig=mu_sig, mu_bg=10)
t_obs, p_obs = nafi.single_ts_and_pvals(lnl_obs, weights, ts=ts, ps=ps)
ll_obs, ul_obs = nafi.intervals(p_obs, hypotheses, cl=0.9)
```

Limitations
============

  1. Nafi is **not a modelling, fitting or simulation framework**. We start from an array of possible likelihoods vs. hypotheses. If you have nuisance parameters, you must take care of choosing appropriate values for the toys, then profiling (or marginalizing) them out in the toy and real data.

        Nafi does include a few example likelihoods, such as the simple (single-bin) Poisson counting experiment in the synopsis. We also have an unbinned rare-event likelihood with signal and background event positions distributed normally, a counting experiment with an uncertain background, etc.

  2. Nafi's **accuracy is limited by the resolution of your hypothesis array**. When `nafi.ts_and_ps` computes the maximum likelihood, it basically takes the maximum of the precomputed values. While `nafi.intervals` interpolates between the p-values of different hypotheses, it does not recompute the likelihood to find the precise point where p = 0.1. In the synopsis example, we also get inaccurate results if the observed n approaches ~50, since we only considered mu_sig up to 42 (with mu_bg = 10).

        This is intentional. Frequentist constructions have to evaluate the likelihood of many toy datasets on a finely spaced grid of hypotheses to accurately capture sharp shifts in the test statistic distribution. Thus, there is little advantage in using an advanced, possibly fragile minimizer to find the best-fit on each toy.
        
        Similarly, recomputing the likelihood to zoom in on the p=0.1 intersection is not enough -- you must also know if and how the test statistic distribution changes. If you need more accuracy, compute new toys on a fine grid near your observed p=0.1 point.

 
