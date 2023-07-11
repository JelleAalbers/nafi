try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    # matplotlib is not a dependency; if you don't have it
    # the plot functions will just crash
    pass

import nafi
import numpy as np

from .methods import FeldmanCousins
from .experiments import Experiment

export, __all__ = nafi.exporter()


@export
def coverage_credibility_plot(
        experiment: Experiment, 
        method_styles: dict = None,
        coverage_limits=(84, 100),
        credibility_limits=(50, 100),
        **common_style):

    xp = experiment
    if method_styles is None:
        method_styles = {
            FeldmanCousins: dict(color='b'),
        }

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
