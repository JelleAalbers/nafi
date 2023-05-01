from functools import partial
import jax
import jax.numpy as jnp

import nafi
export, __all__ = nafi.exporter()


def _logit(p):
    return jnp.log(p/(1-p))


@export
@partial(jax.jit, static_argnames=('n_x', 'x_transform'))
def get_lnl(mu_hyp, n_x=1000, x_transform=_logit):
    """Return (logl, toy weight) for a Gaussian measurement
    of a parameter mu constrained to >= 0.

    That is, the true value is at mu_hyp >= 0, and the experiment observes
        x = Norm(mu_hyp, 1)
    
    Both results are (n_outcomes, hypotheses) arrays:
        lnl contains the log likelihood at each hypothesis,
        toy_weight contains P(outcome | hypotheses), normalized over outcomes

    Arguments:
        mu_hyp: Array with hypotheses
        n_x: Number of x outcomes to consider
        x_transform: Function to transform a uniform 0-1 grid (minus endpoints)
            to x - values to integrate over. Defaults to logit.
    """
    # Possible observations x and their weights compared to uniform
    # (n_outcomes,)
    _p = jnp.linspace(0, 1, n_x + 2)[1:-1]
    x = x_transform(_p)
    dx = jax.vmap(jax.grad(x_transform))(_p)[:,None]

    # Log likelihood
    # (n_outcomes, n_hyp)
    lnl = jax.scipy.stats.norm.logpdf(x[:,None], loc=mu_hyp[None,:])

    # Probability of outcome, given hypothesis
    # (n_outcomes, n_hyp)
    p_x = jnp.exp(lnl) * dx
    toy_weight = p_x / p_x.sum(axis=0)

    return lnl, toy_weight
