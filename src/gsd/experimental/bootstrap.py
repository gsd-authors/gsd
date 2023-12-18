from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from gsd.experimental.fit import Estimator
from .. import GSDParams
from ..gsd import log_prob, sample, sufficient_statistic


def t_statistic(n: ArrayLike, p: ArrayLike) -> Array:
    """
    Calculates the T statistic (as defined by BĆ for the G-test)

    :param n: counts of observations in each cell (can be an array with dimensions num_samples x num_of_cells)
    :param p: expected probabilities of each cell (can be an array with dimensions num_samples x num_of_cells)
    :return: T statistic
    """
    n_total = jnp.sum(n, axis=-1, keepdims=True)

    T = n * jnp.log(n / (n_total * p))
    T = jnp.where(n == 0, 0, T).sum(axis=-1)
    return T


def g_test(n: Array, p: Array, m: Array, q: Array) -> Array:
    """
    G-test,"Bogdan: stosujemy bootstrapową wersję (zamiast asymptotycznej ze względu na małe n) klasycznego testu o
    nazwie G-test czyli testu ilorazu wiarygodności."

    :param n: Observation counts :math:`(n_1, n_2, n_3, n_4, n_5)`, a 1d array
    :param p: Estimated distribution :math:`(p_1, p_2, p_3, p_4, p_5)`, a 1d array
    :param m: T Bootstrap samples from distribution :math:`p`, Array[T,5]
    :param q: T estimated distributions for bootstrapped samples, array[T,5]
    :return: G-test p-value
    """
    n_non_zero_cells = (n != 0).sum()
    if n_non_zero_cells == 1:
        return 1.0

    # Return a p-value of 1.0 only if exactly any two NEIGHBOURING cells are non-zero
    if n_non_zero_cells == 2:
        # Find indices of the top 2 elements
        top_two_idx = np.argpartition(n, -2)[-2:]
        idx_diff = np.abs(top_two_idx[0] - top_two_idx[1])
        # Only if the top 2 elements are neighbours, return 1.0
        if idx_diff == 1:
            return 1.0

    T = jax.jit(t_statistic)(n, p)
    Tr = jax.jit(jax.vmap(t_statistic))(m, q)
    return np.mean(Tr >= T)


def prob(x: GSDParams) -> Array:
    """Compute probabilities of each answer.

    :param x: Parametrs of GSD
    :return: An array of probabilities
    """
    return jnp.exp(jax.vmap(log_prob, in_axes=(None, None, 0))(x.psi, x.rho,
                                                               jnp.arange(1,
                                                                          6)))


class BootstrapResult(NamedTuple):
    probs: Array
    bootstrap_samples: Array
    bootstrap_probs: Array


@partial(jax.jit, static_argnums=(1, 3, 4))
def static_bootstrap(data: ArrayLike, estimator: Estimator, key: Array,
                      n_bootstrap_samples: int,
                      n_total_scores: int) -> BootstrapResult:
    theta_hat = estimator(data)
    exp_prob_gsd = prob(theta_hat)

    bootstrap_samples_gsd = sample(theta_hat.psi, theta_hat.rho,
                                   (n_bootstrap_samples, n_total_scores), key)
    bootstrap_samples_gsd = jax.vmap(sufficient_statistic)(
        bootstrap_samples_gsd)

    bootstrap_fit = jax.lax.map(estimator, bootstrap_samples_gsd)

    bootstrap_exp_prob_gsd = jax.vmap(prob)(bootstrap_fit)
    return BootstrapResult(probs=exp_prob_gsd,
                           bootstrap_samples=bootstrap_samples_gsd,
                           bootstrap_probs=bootstrap_exp_prob_gsd)


def pp_plot_data(data: ArrayLike, estimator: Estimator, key: Array,
                 n_bootstrap_samples: int) -> Array:
    n = int(np.sum(data))
    b = static_bootstrap(data, estimator, key, n_bootstrap_samples, n)
    p_value = g_test(data, b.probs, b.bootstrap_samples, b.bootstrap_probs)
    return p_value
