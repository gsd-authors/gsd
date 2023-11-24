import unittest

import jax.numpy as jnp

import gsd.fit
from src.gsd.fit import log_pmax, pairs, pmax


class FitTestCase(unittest.TestCase):
    def test_mle(self):
        #                 1  2  3 4 5
        data = jnp.asarray([0, 10, 10, 0, 0.])
        _, os = gsd.fit.fit_mle(data)
        self.assertAlmostEqual(os.params.psi, 2.5)
        self.assertAlmostEqual(os.params.rho, 1)

    def test_mle_appendix(self):
        #                 1  2  3 4 5
        data = jnp.asarray([0, 2, 2, 0, 0.])
        _, os = gsd.fit.fit_mle(data, constrain_by_pmax=True)
        self.assertNotAlmostEqual(os.params.psi, 2.5)
        self.assertNotAlmostEqual(os.params.rho, 1)

    def test_mle_single(self):
        data = [20,0,0,0,0.]

        params, os = gsd.fit.fit_mle(data, constrain_by_pmax=False)
        self.assertFalse(jnp.isnan(params.psi))
        self.assertFalse(jnp.isnan(params.rho))

        params, os = gsd.fit.fit_mle(data, constrain_by_pmax=True)
        self.assertFalse(jnp.isnan(params.psi))
        self.assertFalse(jnp.isnan(params.rho))


    def test_list(self):
        #     1  2  3 4 5
        data = [0, 10, 10, 0, 0.]
        _, os = gsd.fit.fit_mle(data)
        self.assertAlmostEqual(os.params.psi, 2.5)
        self.assertAlmostEqual(os.params.rho, 1)

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


if __name__ == '__main__':
    unittest.main()
