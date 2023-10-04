from copy import copy
import dataclasses
import typing

import nafi
from scipy import stats

export, __all__ = nafi.exporter()


@export
class DataAboutIntervals(typing.NamedTuple):
    p_outcome: dict
    credibility: nafi.jax_or_np_array
    brazil: dict
    brazil_ll: dict


@export
@dataclasses.dataclass
class Experiment:

    lnl: nafi.jax_or_np_array
    weights: nafi.jax_or_np_array
    hypotheses: nafi.jax_or_np_array

    cl: float = 0.9
    singular_is_empty: bool = False

    # Useful for plotting, e.g. coverage plots
    name: str = ""
    hypothesis_label: str = ""
    hypothesis_plot_range: tuple = (None, None)

    # In case the outcomes can be summarized usefully by a single number,
    # the attributes are useful for plotting.
    # (e.g. scalar experiments, experiments with sufficient statistics)
    discrete_outcomes: bool = False
    outcome_label: str = ""
    outcomes: nafi.jax_or_np_array = None
    outcome_plot_range: tuple = (None, None)

    # Optional, for a few uncommon variations on CLs
    expected_events: nafi.jax_or_np_array = None

    def __post_init__(self):
        self.posterior_cdf = nafi.posterior_cdf(
            nafi.posterior(self.lnl, self.hypotheses))
        # p of background-only hypothesis
        # TODO: check, why do we need the 1-???
        # TODO: can be done easier, don't need to compute q0 for all hyps...
        _, p0 = nafi.ts_and_pvals(self.lnl, self.weights, statistic='q0')
        self.p0 = 1 - p0[:,0]
        self.clear_cache()

    def clear_cache(self):
        """Clear cached results and limits."""
        self._limits = dict()
        self._results = dict()

    def clear_copy(self):
        """Return a copy of the experiment with cleared cache."""
        self_copy = copy(self)
        self_copy.clear_cache()
        return self_copy

    def is_discovery(self, min_sigma):
        """Return a bool (n_outcomes,) array indicating whether the outcome
        is a discovery at the given significance level."""
        return self.p0 < stats.norm.cdf(-min_sigma)

    def evaluate_intervals(self, ll, ul):
        return DataAboutIntervals(
            p_outcome = nafi.outcome_probabilities(
                ll, ul,
                weights=self.weights,
                hypotheses=self.hypotheses,
                singular_is_empty=self.singular_is_empty),
            credibility = nafi.credibility(
                self.posterior_cdf,
                self.hypotheses,
                ll, ul),
            brazil = nafi.brazil_band(ul, self.weights),
            brazil_ll = nafi.brazil_band(ll, self.weights)
        )

    def get_intervals(self, method_class):
        # Can't use setdefault, does not short-circuit
        if method_class not in self._limits:
            self._limits[method_class] = method_class().get_intervals(xp=self)
        return self._limits[method_class]

    def get_results(self, method_class):
        if method_class not in self._results:
            self._results[method_class] = self.evaluate_intervals(
                *self.get_intervals(method_class))
        return self._results[method_class]
