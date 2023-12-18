import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gsd.experimental import fit_mle_grid
from gsd.experimental.bootstrap import pp_plot_data
from gsd.experimental.fit import GridEstimator
from gsd.fit import log_pmax, pairs, pmax, GSDParams, fit_moments


class FitTestCase(unittest.TestCase):
    def test_pairs(self):
        a = pairs()
        x = jnp.arange(1, 6)
        p = pmax(x)
        self.assertAlmostEqual(p, 5 + 4)

    def test_log_pmax(self):
        x = jnp.arange(1, 6)
        x = x / x.sum()
        p = pmax(x)
        lpm = log_pmax(jnp.log(x))
        self.assertAlmostEqual(jnp.log(p), lpm, places=4)

    def test_fit_grid(self):
        data = jnp.asarray([20, 0, 0, 0, 0.0])
        theta = fit_mle_grid(data, GSDParams(32, 32), False)
        self.assertAlmostEqual(theta.psi, 1.0)

    def test_fit_grid2(self):
        data = jnp.asarray([7, 19., 0, 0, 0])
        theta = fit_mle_grid(data, GSDParams(128, 64),
                             False)
        self.assertAlmostEqual(theta.rho, 1.0)

    def test_fit_grid3(self):
        num = GSDParams(16, 8)
        est = GridEstimator.make(num)
        data = jnp.asarray([7, 25., 0, 0, 0])
        hat = est(data)
        theta = fit_mle_grid(data, num, False)
        jax.tree_util.tree_map(lambda a,b: self.assertAlmostEqual(a,b,2), hat, theta)

        ...


class PPTestCase(unittest.TestCase):
    def test_pp(self):
        # @jax.jit
        # def estimator(x):
        #     if len(x.shape) > 1:
        #         return jax.vmap(fit_moments)(x)
        #     return fit_moments(x)

        data = jnp.asarray([10, 5, 1, 0, 0.0])
        n = int(np.sum(data))
        p_val = pp_plot_data(data, fit_moments, jax.random.key(42), 99)
