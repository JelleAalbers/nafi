from copy import copy
import dataclasses
import typing as ty

import nafi
import numpy as np
from scipy import stats

export, __all__ = nafi.exporter()


@export
class DataAboutIntervals(ty.NamedTuple):
    p_outcome: dict
    credibility: nafi.jax_or_np_array
    brazil: dict
    brazil_ll: dict


@export
@dataclasses.dataclass(kw_only=True)
class ExperimentBase:

    hypotheses: nafi.jax_or_np_array

    cl: float = 0.9
    singular_is_empty: bool = False

    # Useful for plotting, e.g. coverage plots
    name: str = ""
    hypothesis_label: str = ""
    hypothesis_plot_range: tuple = (None, None)
    hypothesis_plot_log: bool = True

    # In case the outcomes can be summarized usefully by a single number,
    # these attributes are useful for plotting.
    # (e.g. scalar experiments, experiments with sufficient statistics)
    discrete_outcomes: bool = False
    outcome_label: str = ""
    outcomes: nafi.jax_or_np_array = None
    outcome_plot_range: tuple = (None, None)
    outcome_plot_log: bool = True

    # Optional, for a few uncommon variations on CLs
    expected_events: nafi.jax_or_np_array = None

    # Whether to evaluate the credibility of the intervals, i.e. fill the
    # DataAboutIntervals.credibility attribute.
    evaluate_credibility: bool = False


@export
@dataclasses.dataclass(kw_only=True)
class Experiment(ExperimentBase):

    lnl: nafi.jax_or_np_array
    weights: nafi.jax_or_np_array

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
        if self.evaluate_credibility:
            credibility = nafi.credibility(
                self.posterior_cdf,
                self.hypotheses,
                ll, ul)
        else:
            credibility = None
        return DataAboutIntervals(
            p_outcome = nafi.outcome_probabilities(
                ll, ul,
                weights=self.weights,
                hypotheses=self.hypotheses,
                singular_is_empty=self.singular_is_empty),
            credibility = credibility,
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


@export
@dataclasses.dataclass(kw_only=True)
class ChangingExperiment(ExperimentBase):
    """
    For studying the influence of a parameter value on the intervals.

    We have to know how to compute the (lnl, weights) for each parameter value.
    and when we store results, keep track of which belongs to which parameter
    value.
    """

    def lnl_and_weights(self, parameter_value):
        raise NotImplementedError

    method_classes: ty.Sequence

    parameter_values: nafi.jax_or_np_array
    parameter_label: str = ""
    parameter_plot_range: tuple = (None, None)
    parameter_plot_log: bool = True

    progress_bar: ty.Callable = lambda x: x

    #: Whether to store the limit for each possible outcome.
    #: Only makes sense if the outcomes are fixed for each parameter value.
    store_limits = False

    def __post_init__(self):
        if self.store_limits:
            self._limits = dict()
        self._results = dict()
        for parameter_value in self.progress_bar(self.parameter_values):
            xp = self.get_experiment(parameter_value)
            for method_class in self.method_classes:
                cache_key = (method_class, parameter_value)
                if self.store_limits:
                    self._limits[cache_key] = xp.get_intervals(method_class)
                self._results[cache_key] = xp.get_results(method_class)

    def get_experiment(self, parameter_value) -> Experiment:
        """Return an Experiment object for the given parameter value."""
        lnl, weights = self.lnl_and_weights(parameter_value)
        field_names = [x.name for x in dataclasses.fields(Experiment)]
        kwargs = dict(lnl=lnl, weights=weights)
        for x in field_names:
            if hasattr(self, x):
                kwargs[x] = getattr(self, x)
        return Experiment(**kwargs)

    def get_limits(self, method_class):
        """Return (lower limits, upper limits) for the method_class,
        each an an array of (n_parameter_values, n_outcomes).

        Makes no sense if the outcomes are incommensurate between parameters.
        """
        if not self.store_limits:
            raise ValueError("Set store_limits = True on initialization")
        # List of (lls, uls) for each parameter value
        results = [
            self._limits[(method_class, parameter_value)]
            for parameter_value in self.parameter_values
        ]
        # Stack each of the lls, uls into a single array
        lls, uls = zip(*results)
        return np.stack(lls), np.stack(uls)

    def get_results(self, method_class):
        """Returns DataAboutIntervals for method_class, where all arrays are
        (n_parameter_values, ...) arrays."""
        # List of DataAboutIntervals for each parameter value
        result_list = [
            self._results[(method_class, parameter_value)]
            for parameter_value in self.parameter_values
        ]

        if self.evaluate_credibility:
            # Stack the credibility arrays into a single array
            credibility = np.stack([x.credibility for x in result_list])
        else:
            credibility = None

        # Stack the other fields arrays into a single array
        return DataAboutIntervals(
            p_outcome = self._stack_dict(result_list, 'p_outcome'),
            credibility = credibility,
            brazil = self._stack_dict(result_list, 'brazil'),
            brazil_ll = self._stack_dict(result_list, 'brazil_ll'),
        )

    def _stack_dict(self, result_list, attr):
        # This will teach me to avoid mixing dicts and namedtuples again...
        return {
            key: np.stack([getattr(x, attr)[key] for x in result_list])
            for key in getattr(result_list[0], attr)
        }
