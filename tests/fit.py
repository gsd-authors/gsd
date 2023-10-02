from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import dblquad

from gsd import log_prob

if __name__ == '__main__':
    data = jnp.asarray([5, 12, 3, 0, 0])
    k = jnp.arange(1, 6)


    @jax.jit
    def posterior(psi, rho):
        log_posterior = jax.vmap(log_prob, in_axes=(None, None, 0))(psi, rho, k) @ data + 1. + 1 / 4.
        posterior = jnp.exp(log_posterior)
        return posterior


    epsabs = 1e-14
    epsreal = 1e-11

    Z, Zerr = dblquad(posterior, a=0, b=1, gfun=lambda x: 1., hfun=lambda x: 5., epsabs=epsabs, epsrel=epsreal)
    psi_hat, _ = dblquad(jax.jit(lambda psi, rho: psi * posterior(psi, rho)), a=0, b=1, gfun=lambda x: 1.,
                         hfun=lambda x: 5.,
                         epsabs=epsabs, epsrel=epsreal)
    psi_hat = psi_hat / Z
    rho_hat, _ = dblquad(jax.jit(lambda psi, rho: rho * posterior(psi, rho)), a=0, b=1, gfun=lambda x: 1.,
                         hfun=lambda x: 5.,
                         epsabs=epsabs, epsrel=epsreal)
    rho_hat = rho_hat / Z

    psi_ci, _ = dblquad(jax.jit(lambda psi, rho: (psi_hat - psi) ** 2 * posterior(psi, rho)), a=0, b=1,
                        gfun=lambda x: 1., hfun=lambda x: 5.,
                        epsabs=epsabs, epsrel=epsreal)

    psi_ci = np.sqrt(psi_ci / Z)

    rho_ci, _ = dblquad(jax.jit(lambda psi, rho: (rho_hat - rho) ** 2 * posterior(psi, rho)), a=0, b=1,
                        gfun=lambda x: 1., hfun=lambda x: 5.,
                        epsabs=epsabs, epsrel=epsreal)

    rho_ci = np.sqrt(rho_ci / Z)

    k @ data / data.sum()
    pass
