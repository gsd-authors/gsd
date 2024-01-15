import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float, Int, PRNGKeyArray

import gsd
from gsd import GSDParams
from gsd.gsd import vmin


@jax.jit
def vmax(mean: Array, N: Int) -> Array:
    """
    Computes maximal variance for categorical distribution supported on Z[1,N]
    :param mean:
    :param N:
    :return:
    """
    return (mean - 1.0) * (N - mean)


def _lagrange_log_probs(lagrage: tuple, dist: 'MaxEntropyGSD'):
    lamda1, lamdam, lamdas = lagrage
    lp = lamda1 + dist.support * lamdam + lamdas * dist.squred_diff - 1.0
    return lp


def _implicit_log_probs(lagrage: tuple, d: 'MaxEntropyGSD'):
    lp = _lagrange_log_probs(lagrage, d)
    p = jnp.exp(lp)
    return (jnp.sum(p) - 1.0,  # jax.nn.logsumexp(lp),
            jnp.dot(p, d.support) - d.mean,
            # jax.nn.logsumexp(a=lp, b=d.support) - jnp.log(d.mean),
            jnp.dot(p, d.squred_diff) - d.sigma ** 2,
            # jax.nn.logsumexp(a=lp, b=d.squred_diff) - 2 * jnp.log(d.sigma)
            )


def _explicit_log_probs(dist: 'MaxEntropyGSD'):
    solver = optx.Newton(rtol=1e-8, atol=1e-8, )

    lgr = jax.tree_util.tree_map(jnp.asarray, (-0.01, -0.01, -0.01))
    sol = optx.root_find(_implicit_log_probs, solver, lgr, args=dist,
                         max_steps=int(1e4), throw=False)
    return _lagrange_log_probs(sol.value, dist)


class MaxEntropyGSD(eqx.Module):
    r"""
    Maximum entropy distribution supported on `Z[1,N]`

    This distribution is defined to fulfill the following conditions on $p_i$

    * Maximize $H= -\sum_i p_i\log(p_i)$ wrt.
    * $\sum p_i=1$
    * $\sum i p_i= \mu$
    * $\sum (i-\mu)^2 p_i= \sigma^2$

    :param mean: Expectation value of the distribution.
    :param sigma: Standard deviation of the distribution.
    :param N: Number of responses

    """
    mean: Float[Array, ""]
    sigma: Float[Array, ""]  # std
    N: int = eqx.field(static=True)


    def log_prob(self, x: Int[Array, ""]):
        lp = _explicit_log_probs(self)
        return lp[x - 1]

    def prob(self, x: Int[Array, ""]):
        return jnp.exp(self.log_prob(x))

    @property
    def support(self):
        return jnp.arange(1, self.N + 1)

    @property
    def squred_diff(self):
        return jnp.square((self.support - self.mean))

    def stddev(self):
        return jnp.sqrt(self.variance())

    def vmax(self):
        return (self.mean - 1.0) * (self.N - self.mean)

    def vmin(self):
        return vmin(self.mean)

    @property
    def all_log_probs(self):
        lp = _explicit_log_probs(self)
        return lp

    @jax.jit
    def entropy(self):
        lp = self.all_log_probs
        return -jnp.dot(lp, jnp.exp(lp))

    def sample(self, key: PRNGKeyArray, axis=-1, shape=None):
        lp = self.all_log_probs
        return jax.random.categorical(key, lp, axis, shape) + self.support[0]

    @staticmethod
    def from_gsd(theta:GSDParams, N:int) -> 'MaxEntropyGSD':
        """Created maxentropy from GSD parameters.

        :param theta: Parameters of a GSD distribution.
        :param N: Support size
        :return: A distribution object
        """
        return MaxEntropyGSD(
            mean=gsd.mean(theta.psi, theta.rho),
            sigma=jnp.sqrt(gsd.variance(theta.psi, theta.rho)),
            N=N
        )

MaxEntropyGSD.__init__.__doc__ = """Creates a MaxEntropyGSD

        :param mean: Expectation value of the distribution.
        :param sigma: Standard deviation of the distribution.
        :param N: Number of responses
        
        .. note::
            An alternative way to construct this distribution is by use of 
            :ref:`from_gsd`
  
"""