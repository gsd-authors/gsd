from jax import config
config.update("jax_enable_x64", True)

import unittest
from math import exp

import gsd
import jax
import jax.tree_util as tree
import jax.numpy as jnp

class JaxGSDTestCase(unittest.TestCase):

    def test_something(self):
        for psi in [1., 1.1, 2., 3., 4., 5.]:
            for rho in [0.0, 0.5,0.8, 1.]:
                for k in range(1, 6):
                    pref = gsd.gsd_prob(psi, rho, k)
                    lpjax = gsd.log_prob(psi, rho, k)
                    print(psi,rho,k, pref,exp(lpjax))
                    self.assertAlmostEqual(pref,exp(lpjax))
                    ...


class AutogradTestCase(unittest.TestCase):
    def test_hessain(self):

        psi=4.
        rho=0.8
        t = (psi, rho)
        def f(t):
            psi, rho = t
            return gsd.log_prob(psi, rho, 2)

        f(t)

        g = jax.grad(f)(t)
        h = jax.hessian(f)(t)

        x = (g,h)
        x = tree.tree_map(jnp.isnan, x)
        x = tree.tree_map(bool, x)


        self.assertTrue(h[0])
        self.assertFalse(any(tree.tree_leaves(x)))

class FancyMathTestCase(unittest.TestCase):
    def test_prod(self):
        from gsd.ref_prob import 𝚷,ℤ

        x = 𝚷(i for i in ℤ[1,3])
        self.assertEqual(x,6)

if __name__ == '__main__':
    unittest.main()
