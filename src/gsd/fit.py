import itertools as it
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jax.typing import ArrayLike

from .gsd import log_prob, vmax, vmin


class GSDParams(NamedTuple):
    """NamedTuple representing parameters for the Generalized Score
    Distribution (GSD).

    This class is used to store the psi and  rho parameters for the GSD. It
    provides a convenient way to group these parameters together for use in
    various statistical and modeling applications.
    """

    psi: Array
    rho: Array


def pairs(M: int = 5) -> Array:
    comb = it.combinations(range(0, M), 2)
    a = jnp.asarray(list(comb))
    return a


@jax.jit
def pmax(probs: Array, M: int = 5) -> Array:
    i = pairs(M)
    sums = jnp.sum(probs[i], axis=1)
    return jnp.max(sums)


# @jax.jit
def log_pmax(log_probs: Array, M: int = 5) -> Array:
    i = pairs(M)
    lsums = jax.scipy.special.logsumexp(log_probs[i], axis=1)
    return jnp.max(lsums)


def allowed_region(log_probs: Array, n: Array, M: int = 5) -> Array:
    # pmax <= 1-1/n
    # logpmax <= log(1-1/n)
    return log_pmax(log_probs, M) <= jnp.log1p(-1.0 / n)


@jax.jit
def fit_moments(data: ArrayLike) -> GSDParams:
    """Fits GSD using moments estimator

    :param data: An Array of counts of each response.
    :return: GSD Parameters
    """

    data = jnp.asarray(data)
    psi = jnp.dot(data, jnp.arange(1, 6)) / jnp.sum(data)
    V = jnp.dot(data, jnp.arange(1, 6) ** 2) / jnp.sum(data) - psi ** 2
    vma = vmax(psi)
    vmi = vmin(psi)
    rho = jnp.where(jnp.allclose(vma,vmi), 0.5, (vmax(psi) - V) / ( vma-vmi) )
    return GSDParams(psi=psi, rho=rho)


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

def make_logits(theta: GSDParams) -> Array:
    logits = jax.vmap(log_prob, (None, None, 0))(theta.psi, theta.rho,
                                                    jnp.arange(1, 6))
    return logits


@partial(jax.jit, static_argnums=[1,2,3,4,5])
def fit_mle(data: ArrayLike, max_iterations: int = 100,
            log_lr_min: ArrayLike = -15, log_lr_max: ArrayLike = 2.0,
            num_lr: ArrayLike = 10, constrain_by_pmax=False, ) -> tuple[
    GSDParams, OptState]:
    """Finds the maximum likelihood estimator of the GSD parameters. The
    algorithm used here is a simple gradient ascent. We use the concept of
    projected gradient to enforce constraints for parameters (psi in [1, 5],
    rho in [0, 1]) and exhaustive search for line search along the gradient.

    :param data: An array of counts for each response. :param
    max_iterations: Maximum number of iterations. :param log_lr_min: Log2 of
    the smallest learning rate. :param log_lr_max: Log2 of the largest
    learning rate. :param num_lr: Number of learning rates to check during
    the line search. :param constrain_by_pmax: Bool flag whether for add
    constrain described in Appendix D

    :return: An opt state whore params filed contains estimated values of
    GSD Parameters
    """

    data = jnp.asarray(data)



    def ll(theta: GSDParams) -> Array:
        logits = make_logits(theta)
        return jnp.dot(data, logits) / jnp.sum(data)

    grad_ll = jax.grad(ll)

    if constrain_by_pmax :
        theta0 = GSDParams(psi=3.0, rho=0.5)
    else:
        theta0 = fit_moments(data)


    rate = jnp.concatenate([jnp.zeros((1,)),
                            jnp.logspace(log_lr_min, log_lr_max, num_lr,
                                         base=2.0)])

    def update(tg, t, lo, hi):
        """
        :param tg: gradient
        :param t: theta parameters
        :param lo: lower bound
        :param hi: upper bound
        :return: updated params
        """
        nt = t + rate * jnp.where(jnp.isnan(tg), 0.0, tg)
        _nt = jnp.where(nt < lo, lo, nt)
        _nt = jnp.where(_nt > hi, hi, _nt)
        return _nt

    lo = GSDParams(psi=1.0, rho=0.0)
    hi = GSDParams(psi=5.0, rho=1.0)

    def body_fun(state: OptState) -> OptState:
        t, _, count = state
        g = grad_ll(t)
        jax.debug.print("grad {0} {1}", *g)
        new_theta = jtu.tree_map(update, g, t, lo, hi)
        new_lls = jax.vmap(ll)(new_theta)
        # filter nan
        new_lls = jnp.where(jnp.isnan(new_lls), -jnp.inf, new_lls)
        # filter pmax
        if constrain_by_pmax:
            n = jnp.sum(data)
            logits = jax.vmap(make_logits)(new_theta)
            is_in_region = jax.vmap(allowed_region, in_axes=(0, None))(logits,
                                                                       n)
            # jax.debug.print("in region {is_in_region}",
            # is_in_region=is_in_region)
            new_lls = jnp.where(is_in_region, new_lls, -jnp.inf)

        max_idx = jnp.argmax(new_lls)
        #jax.debug.print("{max_idx}||| {new_lls}",max_idx=max_idx,new_lls=new_lls)
        ret= OptState(params=jtu.tree_map(lambda t: t[max_idx], new_theta),
                        previous_params=t, count=count + 1, )
        #jax.debug.print("body: {0} {1}",*ret.params)
        return ret

    def cond_fun(state: OptState) -> Array:
        tn, tnm1, c = state
        should_stop = jnp.logical_or(jnp.all(jnp.array(tn) == jnp.array(tnm1)),
                                     c > max_iterations)
        # stop on NaN
        should_stop = jnp.logical_or(should_stop,
                                     jnp.any(jnp.isnan(jnp.array(tn))))
        return jnp.logical_not(should_stop)

    opt_state = jax.lax.while_loop(cond_fun, body_fun,
                                   OptState(params=theta0,
                                            previous_params=jtu.tree_map(
                                                lambda _: jnp.inf, theta0),
                                            count=0, ), )
    return opt_state.params, opt_state
