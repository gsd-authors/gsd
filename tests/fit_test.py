import unittest
import jax.numpy as jnp

import gsd.fit


class FitTestCase(unittest.TestCase):
    def test_mle(self):
        #                 1  2  3 4 5
        data=jnp.asarray([0,10,10,0,0.])
        _,os = gsd.fit.fit_mle(data)
        self.assertAlmostEqual(os.params.psi, 2.5)
        self.assertAlmostEqual(os.params.rho, 1)



if __name__ == '__main__':
    unittest.main()
