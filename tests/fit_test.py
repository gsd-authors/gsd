import unittest

import jax.numpy as jnp

import gsd.experimental
import gsd.fit



class FitTestCase(unittest.TestCase):
    def test_mle(self):
        #                 1  2  3 4 5
        data = jnp.asarray([0, 10, 10, 0, 0.])
        _, os = gsd.experimental.fit_mle(data)
        self.assertAlmostEqual(os.params.psi, 2.5)
        self.assertAlmostEqual(os.params.rho, 1)

    # def test_mle_appendix(self):
    #     #                 1  2  3 4 5
    #     data = jnp.asarray([0, 2, 2, 0, 0.])
    #     _, os = gsd.experimental.fit_mle(data, constrain_by_pmax=True)
    #     self.assertNotAlmostEqual(os.params.psi, 2.5)
    #     self.assertNotAlmostEqual(os.params.rho, 1)
    #
    # def test_mle_single(self):
    #     data = [20,0,0,0,0.]
    #
    #     params, os = gsd.experimental.fit_mle(data, constrain_by_pmax=False)
    #     self.assertFalse(jnp.isnan(params.psi))
    #     self.assertFalse(jnp.isnan(params.rho))
    #
    #     params, os = gsd.experimental.fit_mle(data, constrain_by_pmax=True)
    #     self.assertFalse(jnp.isnan(params.psi))
    #     self.assertFalse(jnp.isnan(params.rho))


    def test_list(self):
        #     1  2  3 4 5
        data = [0, 10, 10, 0, 0.]
        _, os = gsd.experimental.fit_mle(data)
        self.assertAlmostEqual(os.params.psi, 2.5)
        self.assertAlmostEqual(os.params.rho, 1)



if __name__ == '__main__':
    unittest.main()
