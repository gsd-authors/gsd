from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import betaln
from jax.typing import ArrayLike


Shape = Sequence[int]

N = 5


def logbinom(n: ArrayLike, k: ArrayLike) -> Array:
    """Stable log of `n choose k`"""
    return -jnp.log1p(n) - betaln(n - k + 1.0, k + 1.0)


def vmin(psi: ArrayLike) -> Array:
    """Compute the minimal possible variance for categorical distribution
    supported on Z[1,N] for a give mean

    :param psi: mean
    :return: variance
    """
    return (jnp.ceil(psi) - psi) * (psi - jnp.floor(psi))


def vmax(psi: ArrayLike) -> Array:
    """Compute the maximal possible variance for categorical distribution
    supported on Z[1,N] for give mean

    :param psi: mean
    :return: variance
    """
    return (psi - 1.0) * (N - psi)


def _C(Vmax: ArrayLike, Vmin: ArrayLike) -> Array:
    return jnp.where(Vmax != Vmin, 3.0 / 4.0 * Vmax / (Vmax - Vmin), 1.0)


def log_prob(psi: ArrayLike, rho: ArrayLike, k: ArrayLike) -> Array:
    """Compute log probability of the response k for given parameters.

    :param psi: mean
    :param rho: dispersion
    :param k: response
    :return: log of the probability in GSD distribution
    """

    index = jnp.arange(0, 6)
    almost_neg_inf = np.log(1e-10)

    Vmin = vmin(psi)
    Vmax = vmax(psi)
    C = _C(Vmax, Vmin)
    beta_bin_part = (
        logbinom(4, k - 1.0)
        + jnp.sum(
            jnp.where(
                index <= k - 2, jnp.log((psi - 1) * rho / 4 + index * (C - rho)), 0.0
            )
        )
        + jnp.sum(
            jnp.where(
                index <= 4 - k, jnp.log((5.0 - psi) * rho / 4 + index * (C - rho)), 0.0
            )
        )
        - jnp.sum(jnp.where(index <= 3, jnp.log(rho + index * (C - rho)), 0.0))
    )

    b0 = jnp.log(jnp.zeros_like(index))
    b0 = b0.at[0].set(jnp.log((5.0 - psi) / 4.0))
    b0 = b0.at[4].set(jnp.log((psi - 1.0) / 4.0))
    beta_bin_part = jnp.where(rho == 0.0, b0[k - 1], beta_bin_part)

    min_var_part = jax.nn.relu(1.0 - jnp.abs(k - psi))
    log_min_var_part = (
        jnp.where(rho < C, 0.0, jnp.log(rho - C))
        - jnp.log1p(-C)
        + jnp.log(min_var_part)
    )
    log_bin_part = (
        jnp.log1p(-rho)
        - jnp.log1p(-C)
        + logbinom(4, k - 1.0)
        + (k - 1) * (jnp.log(psi - 1) - jnp.log(4))
        + (5 - k) * (jnp.log(5 - psi) - jnp.log(4))
    )

    logmix = jnp.logaddexp(
        jnp.where(min_var_part == 0, almost_neg_inf, log_min_var_part), log_bin_part
    )

    logmix = jnp.where(rho == 1.0, jnp.log(min_var_part), logmix)
    # logmix = jnp.where(min_var_part == 0, log_bin_part, logmix)

    return jnp.where(rho < C, beta_bin_part, logmix)


def mean(psi: ArrayLike, rho: ArrayLike) -> Array:
    """Mean of GSD distribution"""
    del rho
    return psi


def variance(psi: ArrayLike, rho: ArrayLike) -> Array:
    """Variance of GSD distribution"""
    return rho * vmin(psi) + (1 - rho) * vmax(psi)


def sample(psi: ArrayLike, rho: ArrayLike, shape: Shape, key: Array) -> Array:
    """Sample from GSD

    :param psi: mean
    :param rho: dispersion
    :param shape: sample shape
    :param key: random key
    :return: Array of shape :param shape:
    """
    index = jnp.arange(1, N + 1)
    logits = jax.vmap(log_prob, in_axes=(None, None, 0))(psi, rho, index)
    return jax.random.categorical(key, logits, shape=shape) + 1


@jax.jit
def sufficient_statistic(data: ArrayLike) -> Array:
    """Compute GSD sufficient statistic from samples.

    :param data: Samples from GSD data[i] in [1..5]
    :return: Counts of each possible value
    """
    bins = jnp.arange(0.5, N + 1.5, 1.0)
    c, _ = jnp.histogram(jnp.asarray(data), bins=bins)
    return c


def softvmin_poly(x: Array, c: float, d: float) -> Array:
    """Smooths approximation to `vmin` function.

    :param x: An argument, this would be psi
    :param d: Cut point of approximation from `[0,0.5)`
    :return: An approximated value `x` such that `abs(round(x)-x)<=d`
    """
    sq1 = jnp.square(x - c)
    sq2 = jnp.square(sq1)

    return (3 * d) / 8 - ((-3 + 4 * d) * sq1) / (4 * d) - sq2 / (8 * d**3)


def make_softvmin(d: float) -> Callable[[Array], Array]:
    """Create a soft approximation to `vmin` function.

    :param d: Cut point of approximation from `[0,0.5)`
    :return: A callable returning n approximated value `vmin` for `x`
     `abs(round(x)-x)<=d`
    """

    def sofvmin(psi: ArrayLike):
        psi = jnp.asarray(psi)
        c = jax.lax.stop_gradient(jnp.round(psi))
        return jnp.where(jnp.abs(psi - c) < d, softvmin_poly(psi, c, d), vmin(psi))

    return sofvmin
