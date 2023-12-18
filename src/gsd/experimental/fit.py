from functools import partial
from typing import NamedTuple, Callable

import jax
import numpy as np
from jax import numpy as jnp, Array, tree_util as jtu
from jax._src.basearray import ArrayLike

from gsd import GSDParams, fit_moments
from gsd.fit import make_logits, allowed_region

Estimator = Callable[[Array], GSDParams]


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


@partial(jax.jit, static_argnums=[1, 2, 3, 4, 5])
def fit_mle(data: ArrayLike, max_iterations: int = 100,
            log_lr_min: ArrayLike = -15, log_lr_max: ArrayLike = 2.0,
            num_lr: ArrayLike = 10, constrain_by_pmax=False, ) -> tuple[
    GSDParams, OptState]:
    """Finds the maximum likelihood estimator of the GSD parameters. The
    algorithm used here is a simple gradient ascent. We use the concept of
    projected gradient to enforce constraints for parameters (psi in [1, 5],
    rho in [0, 1]) and exhaustive search for line search along the gradient.

    *Since the mass function is not smooth, a gradient-based estimator can fail*

    :param data: An array of counts for each response.
    :param max_iterations: Maximum number of iterations.
    :param log_lr_min: Log2 of
    the smallest learning rate.
    :param log_lr_max: Log2 of the largest
    learning rate.
    :param num_lr: Number of learning rates to check during
    the line search.
   
    :return: An opt state whore params filed contains estimated values of
    GSD Parameters
    """

    data = jnp.asarray(data)

    def ll(theta: GSDParams) -> Array:
        logits = make_logits(theta)
        return jnp.dot(data, logits) / jnp.sum(data)

    grad_ll = jax.grad(ll)

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
        _nt = jnp.clip(nt, lo, hi)
        return _nt

    lo = GSDParams(psi=1.0, rho=0.0)
    hi = GSDParams(psi=5.0, rho=1.0)

    def body_fun(state: OptState) -> OptState:
        t, _, count = state
        g = grad_ll(t)
        # jax.debug.print("grad {0} {1}", *g)
        new_theta = jtu.tree_map(update, g, t, lo, hi)
        new_lls = jax.vmap(ll)(new_theta)
        # filter nan
        new_lls = jnp.where(jnp.isnan(new_lls), -jnp.inf, new_lls)
        max_idx = jnp.argmax(new_lls)
        # jax.debug.print("{max_idx}||| {new_lls}",max_idx=max_idx,new_lls=new_lls)
        ret = OptState(params=jtu.tree_map(lambda t: t[max_idx], new_theta),
                       previous_params=t, count=count + 1, )
        # jax.debug.print("body: {0} {1}",*ret.params)
        return ret

    def cond_fun(state: OptState) -> Array:
        tn, tnm1, c = state
        should_stop = jnp.logical_or(jnp.all(jnp.array(tn) == jnp.array(tnm1)),
                                     c > max_iterations)
        # stop on NaN
        should_stop = jnp.logical_or(should_stop,
                                     jnp.any(jnp.isnan(jnp.array(tn))))
        return jnp.logical_not(should_stop)

    opt_state = jax.lax.while_loop(cond_fun, body_fun, OptState(params=theta0,
                                                                previous_params=jtu.tree_map(
                                                                    lambda
                                                                        _: jnp.inf,
                                                                    theta0),
                                                                count=0, ), )
    return opt_state.params, opt_state


def _make_map(psis, rhos, n):
    f = lambda psi, rho: allowed_region(
        make_logits(GSDParams(psi=psi, rho=rho)), n)
    f = jax.vmap(f, in_axes=(0, None))
    f = jax.vmap(f, in_axes=(None, 0))
    return f(psis, rhos)


def fit_mle_grid(data: ArrayLike, num: GSDParams,
                 constrain_by_pmax=False) -> GSDParams:
    """Fit GSD using naive grid search method.
    This function uses `numpy` and cannot be used in `jit`

        >>> data = jnp.asarray([20, 0, 0, 0, 0.0])
        >>> theta = fit_mle_grid(data, GSDParams(32,32),False)


    :param data: An array of counts for each response.
    :param num: Number of search for each parameter
    :param constrain_by_pmax: Bool flag whether add
    constrain described in Appendix D

    :return: Fitted parameters
    """
    lo = GSDParams(psi=1., rho=0.)
    hi = GSDParams(psi=5., rho=1.)

    grid_exes = jtu.tree_map(jnp.linspace, lo, hi, num)

    def ll(psi, rho) -> Array:
        ll = jnp.asarray(data) * make_logits(GSDParams(psi=psi, rho=rho))
        ll = jnp.where(jnp.isnan(ll), 0., ll)
        return jnp.sum(ll)

    grid_ll = jax.vmap(ll, in_axes=(0, None))
    grid_ll = jax.vmap(grid_ll, in_axes=(None, 0), out_axes=1)
    grid_ll = jax.jit(grid_ll)

    lls = grid_ll(grid_exes.psi, grid_exes.rho)

    if constrain_by_pmax:
        tv = jax.jit(_make_map)(grid_exes.psi, grid_exes.rho, data.sum())
        i, j = np.where(tv)
    else:
        tv = jnp.ones_like(lls)
        i, j = np.where(tv)

    k = np.argmax(lls[i, j])
    return GSDParams(psi=grid_exes.psi[i[k]], rho=grid_exes.rho[j[k]])


class GridEstimator(NamedTuple):
    """ Stateful MLE based on grid search

    :param psis: Grid of psi axis
    :param rhos: Grid of rho axis
    :param lps: Grid of `log_prob` for each answer and each entry in the axes.
    """
    psis: Array
    rhos: Array
    lps: Array

    @staticmethod
    def make(num: GSDParams)->"GridEstimator":
        """Make a grid estimator for GSD. This estimator precomputed log
        probabilities for each answer on a regular grid.

        :param num: Number of grid points

        :return: Estimator
        """
        lo = GSDParams(psi=1., rho=0.)
        hi = GSDParams(psi=5., rho=1.)

        grid_exes = jtu.tree_map(jnp.linspace, lo, hi, num)

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None))
        @partial(jax.vmap, in_axes=(None, 0))
        def _make_logis(psi, rho):
            logits = make_logits(GSDParams(psi, rho))
            small = -1e12
            logits = jnp.where(logits < small, small, logits)
            return logits  # ,psi,rho

        return GridEstimator(psis=grid_exes.psi, rhos=grid_exes.rho,
            lps=_make_logis(grid_exes.psi, grid_exes.rho))

    @jax.jit
    def __call__(self, data: Array) -> GSDParams:
        """Fit GSD using naive grid search method.
        This is a stateful version of ``fit_mle_grid`` that supports `jax.vmap`
        and `jax.jit`.

        :param data: An array of counts for each response.
        :return: Fitted parameters
        """
        map = self.lps @ data
        flat_idx = jnp.argmax(map)
        i, j = jnp.unravel_index(flat_idx, map.shape)
        hat = GSDParams(psi=self.psis[i], rho=self.rhos[j])
        return hat
