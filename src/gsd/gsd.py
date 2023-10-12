from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.random import PRNGKeyArray
from jax.scipy.special import betaln
from jax.typing import ArrayLike

Shape = Sequence[int]

N = 5


def logbinom(n: ArrayLike, k: ArrayLike) -> Array:
    """ Stable log of `n choose k` """
    return -jnp.log1p(n) - betaln(n - k + 1., k + 1.)


def vmin(psi: ArrayLike) -> Array:
    """Compute the minimal possible variance for give mean

    :param psi: mean
    :return: variance
    """
    return (jnp.ceil(psi) - psi) * (psi - jnp.floor(psi))


def vmax(psi: ArrayLike) -> Array:
    """Compute the maximal possible variance for give mean

    :param psi: mean
    :return: variance
    """
    return (psi - 1.) * (5 - psi)


def _C(Vmax: ArrayLike, Vmin: ArrayLike) -> Array:
    return jnp.where(Vmax != Vmin, 3. / 4. * Vmax / (Vmax - Vmin), 1.)


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
    beta_bin_part = logbinom(4, k - 1.) + jnp.sum(
        jnp.where(index <= k - 2, jnp.log((psi - 1) * rho / 4 + index * (C - rho)), 0.)) + jnp.sum(
        jnp.where(index <= 4 - k, jnp.log((5. - psi) * rho / 4 + index * (C - rho)), 0.)) - jnp.sum(
        jnp.where(index <= 3, jnp.log(rho + index * (C - rho)), 0.))

    b0 = jnp.log(jnp.zeros_like(index))
    b0 = b0.at[0].set(jnp.log((5. - psi) / 4.))
    b0 = b0.at[4].set(jnp.log((psi - 1.) / 4.))
    beta_bin_part = jnp.where(rho == 0.0, b0[k - 1], beta_bin_part)

    min_var_part = jax.nn.relu(1. - jnp.abs(k - psi))
    log_min_var_part = jnp.where(rho < C, 0., jnp.log(rho - C)) - jnp.log1p(-C) + jnp.log(min_var_part)
    log_bin_part = jnp.log1p(-rho) - jnp.log1p(-C) + logbinom(4, k - 1.) + (k - 1) * (jnp.log(psi - 1) - jnp.log(4)) + (
            5 - k) * (jnp.log(5 - psi) - jnp.log(4))

    logmix = jnp.logaddexp(jnp.where(min_var_part == 0, almost_neg_inf, log_min_var_part), log_bin_part)

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


def sample(psi: ArrayLike, rho: ArrayLike, shape: Shape, key: PRNGKeyArray) -> Array:
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
