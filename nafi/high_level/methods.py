import nafi
import jax.numpy as jnp
import numpy as np

from .experiments import Experiment

export, __all__ = nafi.exporter()


# From https://stackoverflow.com/a/13624858
class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def str_maybe_round(x):
    if x == int(x):
        return str(int(x))
    return str(x)

def sigma_to_str(x):
    return "$" + str_maybe_round(x) + " \sigma$"


@export
class IntervalMethod:
    """A method for setting confidence intervals"""

    @classproperty
    def name(cls):
        """The full name of the method"""
        return cls.__name__

    @classproperty
    def short_name(cls):
        """A shorter, but still human-readable name for the method"""
        return cls.name

    @classproperty
    def key(cls):
        """A programmatic identifier of the method"""
        return cls.short_name.lower().replace(' ', '_')


##
# Basic methods
##


@export
class SimpleFrequentist(IntervalMethod):
    statistic: str
    cls = False
    astymptotic = False

    def get_intervals(self, xp: Experiment):
        _, ps = nafi.ts_and_pvals(
            xp.lnl,
            xp.weights,
            statistic=self.statistic,
            cls=self.cls)
        return nafi.intervals(ps, xp.hypotheses, cl=xp.cl)


@export
class FeldmanCousins(SimpleFrequentist):
    name = "Unified / Feldman-Cousins"
    short_name = "FC"

    statistic = 't'


@export
class ClassicUL(SimpleFrequentist):
    name = "Classic upper limits"
    short_name = "UL"

    statistic = 'signedt'


@export
class LEPUL(SimpleFrequentist):
    name = "LEP upper limits"
    short_name = "UL"

    statistic = 'lep'


@export
class ClassicLL(SimpleFrequentist):
    name = "Classic lower limits"
    short_name = "UL"

    # Should maybe just make q_ll...
    statistic = 'minus_signedt'


@export
class CLs(SimpleFrequentist):
    name = "CLs"

    statistic = 'q'
    cls = True


@export
class CLsLEP(SimpleFrequentist):
    name = "CLs LEP"

    statistic = 'lep'
    cls = True


##
# Slightly funky methods
##

@export
class Shortest(IntervalMethod):

    name = "Shortest intervals"
    short_name = "Shortest"

    def get_intervals(self, xp):
        # For each hypothesis, we want to add outcomes in order of raw
        # probability to the Neyman band. The highest lnl should get p=1.
        # neyman_pvals assigns p=1 to the lowest t (0), so we feed it -lnl:
        ps = nafi.neyman_pvals(-xp.lnl, xp.weights)
        return nafi.intervals(ps, xp.hypotheses, cl=xp.cl)


@export
class CentralIntervals(IntervalMethod):
    name = "Central intervals"
    short_name = "Central"

    def get_intervals(self, xp: Experiment):
        # 90% central intervals are made of use 95% lower and upper limits.
        new_cl = 1 - (1 - xp.cl) / 2

        ts, ps = nafi.ts_and_pvals(xp.lnl, xp.weights, statistic='signedt')
        _, ul = nafi.intervals(ps, xp.hypotheses, cl=new_cl)

        # ps of -ts are not just 1 - ps of ts for discrete outcomes,
        # have to redo the Neyman construction.
        ps = nafi.neyman_pvals(-ts, xp.weights)
        ll, _ = nafi.intervals(ps, xp.hypotheses, cl=new_cl)

        # For continous outcomes, represented by a finite sample,
        # this causes ul to always be positive.

        return ll, ul


##
# Bayes methods
##
@export
class BayesUL(IntervalMethod):
    name = "Bayesian upper limit"
    short_name = "Bayes UL"

    def get_intervals(self, xp: Experiment):
        return nafi.intervals(
            # Same as in bayesian.py
            1 - xp.posterior_cdf,
            xp.hypotheses,
            cl=xp.cl)

@export
class BayesHDPI(IntervalMethod):
    name = "Highest-density posterior interval"
    short_name = "Bayes HDPI"

    def get_intervals(self, xp: Experiment):
        ps = nafi.bayesian_pvals(xp.lnl, xp.hypotheses, interval_type='hdpi')
        return nafi.intervals(ps, xp.hypotheses, cl=xp.cl)


##
# Variations on CLs
##

class CLsLike(IntervalMethod):
    def get_intervals(self, xp: Experiment):
        qs = nafi.test_statistics(xp.lnl, statistic='q')
        ps_sb = nafi.neyman_pvals(qs, xp.weights)
        ps_b = nafi.neyman_pvals(qs, xp.weights, freeze_truth_index=0)

        if xp.expected_events is None:
            raise ValueError("Expected events must be set for this method")

        self._limits_from_ps(ps_sb, ps_b, xp)

@export
class JinMcNamara(CLsLike):
    name = "Jin/McNamara method"
    short_name = 'JinMcNamara'

    def _limits_from_ps(self, ps_sb, ps_b, xp):
        return nafi.intervals(
            ps_sb + (1 - ps_b) * np.exp(-xp.expected_events)[None,:],
            xp.mu_sig,
            cl=xp.cl)

@export
class CLClip(CLsLike):
    name = "Upper limits clipped at bg-free value"
    short_name = 'UL clip at bg.free'

    def _limits_from_ps(self, ps_sb, ps_b, xp):
        return nafi.intervals(
            np.maximum(ps_sb, np.exp(-xp.expected_events)[None,:]),
            xp.mu_sig,
            cl=xp.cl)

##
# Discovery threshold
##

@export
class DiscoveryThresholdMethod(IntervalMethod):
    base_method: IntervalMethod
    min_sigma: float

    @classproperty
    def name(cls):
        return f"{cls.base_method.short_name}, {sigma_to_str(cls.min_sigma)} discovery threshold"

    @classproperty
    def short_name(cls):
        return f"{cls.base_method.short_name}, {sigma_to_str(cls.min_sigma)} d.t."

    @classproperty
    def key(cls):
        return f"{cls.base_method.key}_dt{str_maybe_round(cls.min_sigma)}"

    def get_intervals(self, xp: Experiment):
        ll, ul = xp.get_intervals(self.base_method)
        ll = jnp.where(xp.is_discovery(self.min_sigma), ll, 0)
        return ll, ul

@export
class FeldmanCousinsThreeSigmaDisc(DiscoveryThresholdMethod):
    base_method = FeldmanCousins
    min_sigma = 3

FCThreeSigmaDisc = FeldmanCousinsThreeSigmaDisc
__all__ += ['FCThreeSigmaDisc']

@export
class CentralIntervalsThreeSigmaDisc(DiscoveryThresholdMethod):
    base_method = CentralIntervals
    min_sigma = 3


##
# FlipFlop methods
##

@export
class FlipFlopMethod(IntervalMethod):
    min_sigma: float
    method_if_discovery: IntervalMethod
    method_otherwise: CLs

    @classproperty
    def name(cls):
        return (
            cls.method_otherwise.short_name
            + ", flip to "
            + cls.method_if_discovery.short_name
            + " on "
            + sigma_to_str(cls.min_sigma)
            + " discovery")

    @classproperty
    def key(cls):
        return (
            cls.method_otherwise.short_name
            + "_flip_"
            + cls.method_if_discovery.short_name
            + "_"
            + str_maybe_round(cls.min_sigma))

    def get_intervals(self, xp: Experiment):
        ll_disc, ul_disc = xp.get_intervals(self.method_if_discovery)
        ll_nodi, ul_nodi = xp.get_intervals(self.method_otherwise)
        is_discovery = xp.is_discovery(self.min_sigma)
        # (ll_cls is just zeros)
        return (
            jnp.where(is_discovery, ll_disc, ll_nodi),
            jnp.where(is_discovery, ul_disc, ul_nodi)
        )

@export
class CLsFlipFlopThreeSigma(FlipFlopMethod):
    min_sigma = 3
    method_if_discovery = FeldmanCousins
    method_otherwise = CLs


@export
class CLsFlipFlopFiveSigma(FlipFlopMethod):
    min_sigma = 5
    method_if_discovery = FeldmanCousins
    method_otherwise = CLs


##
# Power-constrained limits
##

@export
class PCLMethod:
    base_method: IntervalMethod
    sigma_pcl: float

    @classproperty
    def name(cls):
        return f"{cls.base_method.short_name}, PCL at " + sigma_to_str(cls.sigma_pcl)

    @classproperty
    def key(cls):
        return f"{cls.base_method.short_name}_pcl" + str_maybe_round(cls.sigma_pcl)

    def get_intervals(self, xp: Experiment):
        ll, ul = xp.get_intervals(self.base_method)
        ul = ul.clip(xp.get_results(self.base_method).brazil[self.sigma_pcl][0], None)
        return ll, ul


@export
class ULPCLMinusOne(PCLMethod):
    base_method = ClassicUL
    sigma_pcl = -1

@export
class ULPCLMedian(PCLMethod):
    base_method = ClassicUL
    sigma_pcl = 0

@export
class FeldmanCousinsPCLMinusOne(PCLMethod):
    base_method = FeldmanCousins
    sigma_pcl = -1

@export
class FeldmanCousinsPCLMedian(PCLMethod):
    base_method = FeldmanCousins
    sigma_pcl = 0

@export
class CentralIntervalsPCLMinusOne(PCLMethod):
    base_method = CentralIntervals
    sigma_pcl = -1

@export
class CentralIntervalsPCLMedian(PCLMethod):
    base_method = CentralIntervals
    sigma_pcl = 0

@export
class FCThreeSigmaDiscPCLMinusOne(PCLMethod):
    base_method = FeldmanCousinsThreeSigmaDisc
    sigma_pcl = -1

@export
class FCThreeSigmaDiscPCLMedian(PCLMethod):
    base_method = FeldmanCousinsThreeSigmaDisc
    sigma_pcl = 0

@export
class CIThreeSigmaDiscPCLMinusOne(PCLMethod):
    base_method = CentralIntervalsThreeSigmaDisc
    sigma_pcl = -1

@export
class CIThreeSigmaDiscPCLMedian(PCLMethod):
    base_method = CentralIntervalsThreeSigmaDisc
    sigma_pcl = 0

##
# Methods that raise the upper limit to that of another method if needed
##

@export
class ClippedULMethod:
    base_method: IntervalMethod
    minimum_ul_method: IntervalMethod

    @classproperty
    def name(cls):
        return f"{cls.base_method.short_name}, UL clipped to {cls.minimum_ul_method.short_name}"

    @classproperty
    def key(cls):
        return f"{cls.base_method.short_name}_clipul_{cls.minimum_ul_method.short_name}"

    def get_intervals(self, xp: Experiment):
        ll, ul = xp.get_intervals(self.base_method)
        _, ul_min = xp.get_intervals(self.minimum_ul_method)
        return ll, jnp.maximum(ul, ul_min)


@export
class FeldmanCousinsCLsClipped(ClippedULMethod):
    base_method = FeldmanCousins
    minimum_ul_method = CLs

@export
class CICLsClipped(ClippedULMethod):
    base_method = CentralIntervals
    minimum_ul_method = CLs

@export
class FeldmanCousinsBayesClipped(ClippedULMethod):
    base_method = FeldmanCousins
    minimum_ul_method = BayesUL

@export
class FCThreeSigmaCLsClipped(ClippedULMethod):
    base_method = FeldmanCousinsThreeSigmaDisc
    minimum_ul_method = CLs

@export
class FCThreeSigmaBayesClipped(ClippedULMethod):
    base_method = FeldmanCousinsThreeSigmaDisc
    minimum_ul_method = BayesUL

@export
class CIThreeSigmaCLsClipped(ClippedULMethod):
    base_method = CentralIntervalsThreeSigmaDisc
    minimum_ul_method = CLs
