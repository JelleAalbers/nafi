"""The on-off problem

A two-bin counting experiment where the second bin ('ancilla') has no signal,
but a multiple tau times the background in the first bin.

n events are observed in the main experiment, m in the ancilla.

The main experiment has signal mu_sig and background mu_bg,
the ancilla only has background mu_bg * tau
"""
import jax
from . import twobin

# Outcomes are the same as in the known-background two-bin case
outcomes = twobin.outcomes


@jax.jit
def profile_lnl(mu_sig_hyp, n, m, *, tau):
    """Return (n_outcomes, n_hyp) log profile likelihood.
    
    That is, mu_bg is replaced with its conditional best-fit value
    for each tested hypothesis.
    """
    # Get best-fit background (n_outcomes, n_hyp)
    b_doublehat = conditional_bestfit_bg(
        mu_sig_hyp[None,:], n[:,None], m[:,None], tau=tau)
    
    return _lnl(
        mu_sig_hyp[None,:], n[:,None], m[:,None], mu_bg=b_doublehat, tau=tau)


@jax.jit
def profile_weights(mu_sig_hyp, n, m, n_obs, m_obs, *, tau):
    """Return (n_outcomes, n_hyp) weights of outcomes using the 
    profile construction for the observed outcome n_obs, m_obs.

    That is, the weights are that for toy data with mu_bg set to 
    the conditional best-fit value to the observed data for each hypothesis.
    """
    # Get conditional best-fit background, (hypotheses,) array
    b_doublehat = conditional_bestfit_bg(
        mu_sig_hyp, n_obs, m_obs, tau=tau)
    # Return weights from assuming these backgrounds are true
    return _weights(
        mu_sig_hyp[None,:],
        n[:,None],
        m[:,None],
        mu_bg=b_doublehat[None,:],
        tau=tau)


@jax.jit
def true_weights(mu_sig_hyp, mu_bg_hyp, n, m, *, tau):
    """Return (outcomes, signal hypotheses, background hypotheses) array 
    with weights of outcomes given the true signal and background hypotheses.

    (i.e. P(outcome | sig, bg), normalized to sum to 1 over outcomes)
    
    Arguments:
     - mu_sig_hyp: Array with signal rate hypotheses
     - mu_bg_hyp: Array with background rate hypotheses
     - n, m, tau: same as always
    """
    return _weights(
        mu_sig_hyp[None,:,None], 
        n[:,None,None], 
        m[:,None,None],
        mu_bg=mu_bg_hyp[None,None,:],
        tau=tau)


@jax.jit
def _lnl(mu_sig, n, m, *, mu_bg, tau):
    """Return (n_outcomes, n_hyp) log likelihood for 
    a *given/fixed/known* mu_bg.

    All arguments must be broadcastable. Factorial term is omitted.
    """
    return twobin._lnl(
        mu_sig, n, m, f_sig_1=1, mu_bg_1=mu_bg, mu_bg_2=tau * mu_bg)


@jax.jit
def _weights(mu_sig, n, m, *, mu_bg, tau):
    """Return (n_outcomes, n_hyp) weights of outcomes 
    for a given/fixed/known mu_bg

    All arguments must be broadcastable.
    """
    return twobin._weights(
        mu_sig, n, m, f_sig_1=1, mu_bg_1=mu_bg, mu_bg_2=tau * mu_bg)


@jax.jit
def conditional_bestfit_bg(mu_sig_hyp, n, m, *, tau):
    """Return the conditional best fit background rate for an on-off experiment.
    See lnl_and_weights for details.

    with unknown background.

    Arguments:
        n: Number of observed events
        m: Number of observed events in the ancillary experiment
        mu_hyp: Array with signal rate hypotheses
        tau: Ratio of background rates in the two experiments

    See e.g. Rolke, Lopez, Conrad (2005) arxiv:0403059,
    or Sen, Walker and Woodrofe (2009)
    """
    _q = n + m - (1+tau) * mu_sig_hyp
    root1 = (
        (_q + (_q**2 + 4 * (1 + tau) * m * mu_sig_hyp)**0.5)
        /(2*(1+tau)))
    # Probably this root is never taken. Oh well.
    # root2 = (
    #     (_q - (_q**2 + 4 * (1 + tau) * m * mu_hyp)**0.5)
    #     /(2*(1+tau)))
    return root1
