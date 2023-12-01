import itertools as it
from typing import NamedTuple

import jax
import jax.numpy as jnp
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


def pmax(probs: Array) -> Array:
    """Calculate the maximum of the sum of two probabilities

    :param probs: probabilities array
    :param M: Number of classes
    :return:
    """
    i = pairs(probs.shape[0])
    sums = jnp.sum(probs[i], axis=1)
    return jnp.max(sums)


def log_pmax(log_probs: Array) -> Array:
    """ Calculate the maximum of log of the sum of two probabilities from logarithsms of probabilities

    :param log_probs: logarithsms of probabilities
    :return: Scalar array
    """
    i = pairs(log_probs.shape[0])
    lsums = jax.scipy.special.logsumexp(log_probs[i], axis=1)
    return jnp.max(lsums)


def allowed_region(log_probs: Array, n: Array) -> Array:
    """Compute whether given log_probs satisfy conditions ``pmax <= 1-1/n`` as
    described in Appendix D.
    This is computed in the log domain as ``logpmax <= log(1-1/n)``.

    :param log_probs: logarithsms of probabilities
    :param n: Total number of obserwations
    :return: Binary array
    """

    return log_pmax(log_probs) <= jnp.log1p(-1.0 / n)


@jax.jit
def fit_moments(data: ArrayLike) -> GSDParams:
    """Fits GSD using moment estimator

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


def make_logits(theta: GSDParams) -> Array:
    """Helper function making log probabilities for each answer
    :param theta: GSD parameter
    :return: Array of logits
    """
    logits = jax.vmap(log_prob, (None, None, 0))(theta.psi, theta.rho,
                                                    jnp.arange(1, 6))
    return logits


