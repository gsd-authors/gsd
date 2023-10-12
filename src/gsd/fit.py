from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jax.typing import ArrayLike

from .gsd import vmax, vmin, log_prob


class GSDParams(NamedTuple):
    """NamedTuple representing parameters for the Generalized Structure Distribution (GSD).

    This class is used to store the psi and  rho parameters for the GSD.
    It provides a convenient way to group these parameters together for use in various
    statistical and modeling applications.
    """
    psi: Array
    rho: Array


@jax.jit
def fit_moments(data: ArrayLike) -> GSDParams:
    """Fits GSD using moments estimator

    :param data: A 5d Array of counts of each response.
    :return: GSD Parameters
    """
    psi = jnp.dot(data, jnp.arange(1, 6)) / jnp.sum(data)
    V = jnp.dot(data, jnp.arange(1, 6) ** 2) / jnp.sum(data) - psi ** 2
    return GSDParams(psi=psi, rho=(vmax(psi) - V) / (vmax(psi) - vmin(psi)))


class OptState(NamedTuple):
    """A class representing the state of an optimization process.

    Attributes:
    :param params (GSDParams): The current optimization parameters.
    :param previous_params (GSDParams): The previous optimization parameters.
    :param count (int): An integer count indicating the step or iteration of the optimization process.

    This class is used to store and manage the state of an optimization algorithm, allowing you
    to keep track of the current parameters, previous parameters, and the step count.

    """
    params: GSDParams
    previous_params: GSDParams
    count: int


@jax.jit
def fit_mle(data: ArrayLike, max_iterations: int = 100, log_lr_min: ArrayLike = -15, log_lr_max: ArrayLike = 2.,
            num_lr: ArrayLike = 10) -> tuple[GSDParams, OptState]:
    """Finds the maximum likelihood estimator of the GSD parameters.
    The algorithm used here is a simple gradient ascent.
    We use the concept of projected gradient to enforce constraints for parameters (psi in [1, 5], rho in [0, 1]) and exhaustive search for line search along the gradient.

    :param data: 5D array of counts for each response.
    :param max_iterations: Maximum number of iterations.
    :param log_lr_min: Log2 of the smallest learning rate.
    :param log_lr_max: Log2 of the largest learning rate.
    :param num_lr: Number of learning rates to check during the line search.

    :return: An opt state whore params filed contains estimated values of GSD Parameters
    """

    def ll(theta: GSDParams) -> Array:
        logits = jax.vmap(log_prob, (None, None, 0), (0))(theta.psi, theta.rho, jnp.arange(1, 6))
        return jnp.dot(data, logits) / jnp.sum(data)

    grad_ll = jax.grad(ll)
    theta0 = fit_moments(data)
    rate = jnp.concatenate([jnp.zeros((1,)), jnp.logspace(log_lr_min, log_lr_max, num_lr, base=2.)])

    def update(tg, t, lo, hi):
        '''
        :param tg: gradient
        :param t: theta parameters
        :param lo: lower bound
        :param hi: upper bound
        :return: updated params
        '''
        nt = t + rate * jnp.where(jnp.isnan(tg), 0., tg)
        _nt = jnp.where(nt < lo, lo, nt)
        _nt = jnp.where(_nt > hi, hi, _nt)
        return _nt

    lo = GSDParams(psi=1., rho=0.)
    hi = GSDParams(psi=5., rho=1.)

    def body_fun(state: OptState) -> OptState:
        t, _, count = state
        g = grad_ll(t)
        new_theta = jtu.tree_map(update, g, t, lo, hi)
        new_lls = jax.vmap(ll)(new_theta)
        max_idx = jnp.argmax(jnp.where(jnp.isnan(new_lls), -jnp.inf, new_lls))
        return OptState(params=jtu.tree_map(lambda t: t[max_idx], new_theta), previous_params=t, count=count + 1)

    def cond_fun(state: OptState) -> bool:
        tn, tnm1, c = state
        should_stop = jnp.logical_or(jnp.all(jnp.array(tn) == jnp.array(tnm1)), c > max_iterations)
        # stop on NaN
        should_stop = jnp.logical_or(should_stop, jnp.any(jnp.isnan(jnp.array(tn))))
        return jnp.logical_not(should_stop)

    opt_state = jax.lax.while_loop(cond_fun, body_fun,
                                   OptState(params=theta0, previous_params=jtu.tree_map(lambda _: jnp.inf, theta0),
                                            count=0))
    return opt_state.params, opt_state

