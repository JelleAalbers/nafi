nafi
=====


![Build Status](https://github.com/JelleAalbers/nafi/actions/workflows/pytest.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nafi/badge/?version=latest)](https://nafi.readthedocs.io/en/latest/?badge=latest)

Nafi computesfrequentist p-values and confidence intervals from likelihoods, without asymptotic approximations such as Wilks' theorem.
CLs and simple Bayesian methods are also supported.

[JAX](https://github.com/google/jax) is used to accelerate computations, so nafi will be much faster if you have a GPU.

Synopsis
---------

Unified / Feldman-Cousins* confidence intervals for a counting experiment:

```python
import nafi
import numpy as np

mu_signal = np.arange(0, 42, 0.01)
mu_background = 10

lnl, weights = nafi.likelihoods.counting.lnl_and_weights(
    mu_signal, mu_background)

ts, ps = nafi.ts_and_pvals(lnl, weights)

lower_limits, upper_limits = nafi.intervals(ps, hypotheses, cl=0.9)
```

*: Feldman and Cousins [[PRD](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.3873)/[arXiv](https://arxiv.org/abs/physics/9711021)] also adds a small ad-hoc adjustment to the upper limits for the specific case of a one-bin counting experiment (and not others). To reproduce this, see [this notebook](https://github.com/JelleAalbers/nafi/blob/main/notebooks/feldman_cousins.ipynb). 



Limitations
------------

  1. Nafi is **not a modelling, fitting, simulation, or caching package**. We just take likelihoods and turn them into p-values, limits, etc. Except for the simple examples below, _you_ must provide the likelihood, with nuisance parameters profiled or marginalized out.
        
  2. Nafi works by **bruteforce scanning** over hypotheses you provide. That makes it robust, but also means it needs much memory and computation to achieve high accuracy. For example:

      * When `nafi.ts_and_ps` computes the maximum likelihood, it just finds the maximum over the provided hypotheses. It does not recompute likelihoods in a minimizer loop.
      * While `nafi.intervals` does interpolate between p-values of different hypotheses, it does not recompute the likelihood to find the precise point where the p-value crosses e.g. p = 0.1.
      * In the synopsis example, we also get inaccurate results if the observed n approaches ~50, since we only considered mu_sig up to 42 (with mu_bg = 10).

If you want a more complete and integrated inference framework, you might like [zfit](https://github.com/zfit/zfit)/[hepstats](https://github.com/scikit-hep/hepstats), or [pyhf](https://github.com/scikit-hep/pyhf), or [RooFit](https://root.cern/manual/roofit/)/[RooStats](https://twiki.cern.ch/twiki/bin/view/RooStats/WebHome)/[HistFactory](https://twiki.cern.ch/twiki/bin/view/RooStats/HistFactory).


Example likelihoods
-------------------
Nafi includes some likelihoods for testing and simple applications. These all live under `nafi.likelihoods`, e.g. as `nafi.likelihoods.counting`.

* `counting`: a single-bin Poisson counting experiment with background, as shown in the synopsis.
* `gaussian`: a single measurement of a Gaussian random variable whose mean is known to be positive.
* `twobin`: a two-bin Poisson counting experiment, with configurable signal and background expectations in each bin.
* `unbinned`: for signal and background events with different distributions along a single observable `x`.
* `two_gaussians`: an example of `unbinned`, where signal and background are equal-width Gaussians separated by a parameter `sigma_sep` times that width.

Below are two examples with profile likelihoods, i.e. problems with nuisance parameters eliminating through profiling. In both cases the background rate is the nuisance parameters.

* `onoff`: derived from `twobin`, where the second bin has no signal but a multiple `tau` of the first bin's expected background. Thus the second bin is an ancillary experiment that calibrates the background.
* `counting_uncbg`, a single-bin counting experiment where the background expectation has a Gaussian uncertainty. This may be ill-defined, as backgrounds cannot be negative; `onoff` is a more proper test case for profile likelihoods.
