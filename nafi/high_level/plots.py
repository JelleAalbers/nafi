try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    # matplotlib is not a dependency; if you don't have it
    # the plot functions will just crash
    pass

import nafi
import numpy as np
from scipy import stats

from .methods import FeldmanCousins
from .experiments import Experiment

export, __all__ = nafi.exporter()

DEFAULT_METHOD_STYLES = {
    FeldmanCousins: dict(color='b'),
}


def brazil_hspan(alpha=0.1, yscale=1):
    plt.axhspan(
        *(yscale * stats.norm.cdf([-1, 1])),
        color='g', alpha=alpha, linewidth=0)
    plt.axhspan(
        *(yscale * stats.norm.cdf([-2, 2])),
        color='yellow', alpha=alpha, linewidth=0)


def _get_p_outcome(xp, method, outcome):
    p_outcomes = xp.get_results(method).p_outcome
    if outcome == 'zero_excluded':
        p = p_outcomes['discovery'] + p_outcomes['degenerate']
    else:
        p = p_outcomes[outcome]
    return p


@export
def outcome_plot(
        experiment,
        outcome = 'mistake',
        method_styles: dict = None,
        *,
        invert=False,
        logit=False,
        percent=True,
        ylim=None,
        ylabel=None,
        brazil=False,
        brazil_kwargs=None,
        legend=True,
        legend_kwargs=None,
        ratio_with=None,
        title=True,
        **common_style):
    xp = experiment
    if method_styles is None:
        method_styles = DEFAULT_METHOD_STYLES
    if ylabel is None and not invert:
        ylabel = dict(
            mistake='P(truth excluded)',
            mistake_ul='P(truth above upper limit)',
            mistake_ll='P(truth below lower limit)',
            empty='P(empty interval)',
            zero_excluded='P(0 excluded | hyp. is true)',
            discovery='P(nonempty but 0 excluded | hyp. is true)',
            excl_if_bg='Power, P(hypothesis excluded | 0 is true)',
            excl_if_bg_ul='P(UL excludes hypothesis | 0 is true)',
            excl_if_bg_ll='P(LL excludes hypothesis | 0 is true)',
        ).get(outcome, outcome)
    if logit:
        if ylim is None:
            ylim = (1e-3, 1 - 1e-3)
        plt.yscale('logit')
        if percent:
            # Scaling by x100 doesn't work with logit axis, so
            # instead fake the scaling by changing the tick labels.
            percent = False
            yticks = np.array([.01, .05, .1, .5, .9, .95, .99])
            plt.yticks(yticks)
            ax = plt.gca()
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_yticklabels([str(int(100 * y)) for y in yticks])
    else:
        if percent is None:
            percent = True
        if ylim is None:
            ylim = (0, 1)
    if percent:
        yscale = 100
    else:
        yscale = 1
    ylim = ylim[0] * yscale, ylim[1] * yscale

    for method, style in method_styles.items():
        style = {**common_style, **style}

        p = _get_p_outcome(xp, method, outcome)
        if ratio_with:
            if isinstance(ratio_with, str):
                p /= _get_p_outcome(xp, method, ratio_with)
            else:
                p /= ratio_with

        if invert:
            p = 1 - p

        plt.plot(
            xp.hypotheses,
            p * yscale,
            label=method.name if legend else None,
            **style)

    plt.ylim(*ylim)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlim(*xp.hypothesis_plot_range)
    plt.xlabel(xp.hypothesis_label)

    if legend:
        if legend_kwargs is None:
            legend_kwargs = dict(loc='best')
        plt.legend(**legend_kwargs)

    if brazil:
        if brazil_kwargs is None:
            brazil_kwargs = dict()
        brazil_hspan(**brazil_kwargs, yscale=yscale)

    if title:
        if title is True:
            title = xp.name
        plt.title(title)


@export
def coverage_plot(
        experiment,
        method_styles: dict = None,
        **kwargs):
    kwargs.setdefault('ylim', (.8, 1))
    outcome_plot(experiment, 'mistake', method_styles,
                 invert=True,
                 **kwargs)
    plt.ylabel("Coverage [%], P(truth included)")


@export
def power_plot(
        experiment, method_styles=None,
        logit=True, brazil=True, legend_kwargs=None, legend=True,
        **kwargs):
    outcome_plot(
        experiment, 'excl_if_bg',
        method_styles, linewidth=0.9, logit=logit, brazil=brazil,
        legend_kwargs=legend_kwargs, legend=legend, **kwargs)
    outcome_plot(
        experiment, 'zero_excluded',
        method_styles, linewidth=1.5, logit=logit, brazil=False,
        legend=False, linestyle='--', **kwargs)
    plt.ylabel("Rejection (solid) or Discovery (dashed) power")


@export
def power_vs_mistake(
        xp, method_styles,
        which='both',
        logit=True,
        title=None,
        subplots=True,
        alpha_label=r'$\alpha$, hypothesis is true',
        rejection_power_label=r'$\mathrm{RP}$, background is true',
        **kwargs):
    if which != 'both':
        suffix = '_' + which
    else:
        suffix = ''

    legend_kwargs = kwargs.get('legend_kwargs', dict())
    del kwargs['legend_kwargs']
    legend_kwargs.setdefault('loc', 'upper left')
    legend_kwargs.setdefault('frameon', False)

    true_style = dict(linestyle=':')
    bg_style = dict(linestyle='-')

    if subplots:
        _, axes = plt.subplots(
            nrows=3, ncols=1, sharex=True, figsize=(6,8),
            gridspec_kw=dict(height_ratios=[3,1,1]))
        plt.subplots_adjust(hspace=0)
        plt.sca(axes[0])

    outcome_plot(
        xp, 'mistake' + suffix, method_styles,
        logit=logit, legend=False, title=False,
        **true_style, **kwargs)
    outcome_plot(
        xp, 'excl_if_bg' + suffix, method_styles,
        logit=logit, legend=True, legend_kwargs=legend_kwargs, title=title,
        **bg_style, **kwargs)

    plt.xlabel("Hypothesis, " + plt.gca().get_xlabel())
    plt.ylabel(
        "P(exclude hypothesis)" + (', %' if kwargs.get('percent', True) else ''))

    ax = plt.gca()
    plt.twinx()
    plt.yticks([])
    l1 ,= plt.plot([], [], color='k', **true_style)
    l2, = plt.plot([], [], color='k', **bg_style)
    legend_kwargs['loc'] = 'lower right' if logit else 'center right'
    plt.legend([l1, l2], [alpha_label, rejection_power_label],
               **legend_kwargs)
    plt.sca(ax)

    if subplots:
        ax = axes[1]
        plt.sca(ax)
        outcome_plot(
            xp, 'excl_if_bg' + suffix, method_styles,
            logit=False, legend=False, title=False, percent=False,
            ratio_with='mistake' + suffix,
            **kwargs)
        plt.ylabel(r"$\mathrm{RP} \, / \, \alpha$")
        plt.ylim(0.8, 29)
        #ax.yaxis.tick_right()
        #ax.yaxis.set_label_position("right")
        plt.yscale('log')
        plt.yticks([1, 2, 5, 10, 20])
        #plt.grid(alpha=0.2, linewidth=1, c='k', axis='y')
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

        ax = axes[2]
        plt.sca(ax)
        outcome_plot(
            xp, 'mistake' + suffix, method_styles,
            logit=False, legend=False, title=False, percent=False,
            ratio_with=(1 - xp.cl),
            **kwargs)
        plt.ylabel(r"$\alpha \, / \, \mathrm{nominal}$")
        plt.ylim(0, 1.1)


@export
def coverage_credibility_plot(
        experiment: Experiment,
        method_styles: dict = DEFAULT_METHOD_STYLES,
        coverage_limits=(84, 100),
        credibility_limits=(50, 100),
        **common_style):
    xp = experiment
    if method_styles is None:
        method_styles = DEFAULT_METHOD_STYLES

    _, axes = plt.subplots(
        nrows=2, ncols=2,
        width_ratios=[2,1], height_ratios=[1,2],
        figsize=(6,6))
    neyman_band_ax = axes[1][0]
    coverage_ax = axes[1][1]
    credibility_ax = axes[0][0]
    axes[0][1].axis('off')

    for method, style in method_styles.items():
        ul, ll = xp.get_intervals(method)
        results = xp.get_results(method)
        style = {**common_style, **style}

        if xp.discrete_outcomes:
            outcome_style = dict(drawstyle='steps-post')
        else:
            outcome_style = dict()

        plt.sca(neyman_band_ax)
        plt.plot(xp.outcomes, ul,
                 **style, **outcome_style,
                 label=method.name)
        if not np.all(ll == 0):
            plt.plot(xp.outcomes, ll,
                     **style, **outcome_style)

        plt.sca(coverage_ax)
        plt.plot(
            100 * (1 - results.p_outcome['mistake']),
            xp.hypotheses,
            **style)

        plt.sca(credibility_ax)
        plt.plot(
            xp.outcomes,
            100 * results.credibility,
            **style, **outcome_style)

    plt.sca(neyman_band_ax)
    plt.xlabel(xp.outcome_label)
    plt.ylabel(xp.hypothesis_label)
    plt.xlim(*xp.outcome_plot_range)
    plt.ylim(*xp.hypothesis_plot_range)
    plt.legend(loc='upper left', frameon=True, fontsize=7)

    plt.sca(coverage_ax)
    plt.xlim(*coverage_limits)
    plt.xticks(np.arange(coverage_limits[0] + 1, coverage_limits[1] + 1, 5))
    plt.xlabel("Coverage [%]")
    plt.ylim(*xp.hypothesis_plot_range)
    plt.yticks([])

    plt.sca(credibility_ax)
    plt.xlim(*xp.outcome_plot_range)
    plt.xticks([])
    plt.ylim(*credibility_limits)
    plt.ylabel("Credibility [%]")

    plt.subplots_adjust(wspace=0.07, hspace=0.07)
    plt.suptitle(xp.name, y=0.95)
