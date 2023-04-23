import numpy as np
from scipy import stats
from scipy.special import logit

import nafi
export, __all__ = nafi.exporter()

@export
def get_lnl(mu_hyp, n_x=1000, x_transform=logit):
    """Return (logl, toy weight) for a Gaussian measurement
    of a parameter mu constrained to >= 0.

    The true value is at mu_hyp, and the experiment observes mu_hyp +- 1.

    Both are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_hyp: Array with hypotheses
        n_x: Number of x outcomes to integrate over
        x_transform: Function to transform a uniform 0-1 grid (minus endpoints)
            to x - values to integrate over. Defaults to logit.
    """
    # Possible observations
    # (n_outcomes, n_hyp)
    x = mu_hyp[None,:] + x_transform(np.linspace(0, 1, n_x + 2)[1:-1])[:,None]
    # Log likelihood
    # (n_outcomes, n_hyp)
    lnl = -(x - mu_hyp[None,:])**2 / 2
    # Probability of outcome, given hypothesis
    # (n_outcomes, n_hyp)
    p_x = stats.norm(loc=mu_hyp[None,:]).pdf(x)
    toy_weight = p_x / p_x.sum(axis=0)

    return lnl, toy_weight
