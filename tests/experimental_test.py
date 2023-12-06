import unittest
from gsd.fit import log_pmax, pairs, pmax, GSDParams
from gsd.experimental import fit_mle_grid
import jax.numpy as jnp


class FitTestCase(unittest.TestCase):
    def test_pairs(self):
        a = pairs()
        x = jnp.arange(1, 6)
        p = pmax(x)
        self.assertAlmostEqual(p, 5+4)

    def test_log_pmax(self):
        x = jnp.arange(1, 6)
        x = x/x.sum()
        p = pmax(x)
        lpm = log_pmax(jnp.log(x))
        self.assertAlmostEqual(jnp.log(p), lpm, places=4)

    def test_fit_grid(self):
        data = jnp.asarray([20, 0, 0, 0, 0.0])
        theta = fit_mle_grid(data, GSDParams(32,32),False)
        self.assertAlmostEqual(theta.psi, 1.0)


