import numpy as np
from jax import config
config.update("jax_enable_x64", True)

import unittest
from math import exp

import gsd
import jax
import jax.tree_util as tree
import jax.numpy as jnp

class JaxGSDTestCase(unittest.TestCase):

    def test_pmf(self):
        for psi in [1., 1.1, 2., 3., 4., 5.]:
            for rho in [0.0, 0.5,0.8, 1.]:
                for k in range(1, 6):
                    pref = gsd.gsd_prob(psi, rho, k)
                    lpjax = gsd.log_prob(psi, rho, k)
                    print(psi,rho,k, pref,exp(lpjax))
                    self.assertFalse(np.isnan(lpjax))
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
    def test_log_prob_grad(self):
        for psi in [1., 1.1, 2., 3., 4., 5.]:
            for rho in [0.0, 0.5, 0.8, 1.]:
                for k in range(1, 6):
                    pref = gsd.gsd_prob(psi, rho, k)
                    @jax.grad
                    def g(theta:tuple, k):
                        psi, rho = theta
                        return gsd.log_prob(psi, rho, k)

                    dldt = g((psi,rho), k)
                    h=1e-9
                    # dldp = (log(gsd.gsd_prob(psi+h, rho,k))-log(gsd.gsd_prob(psi, rho,k)))/h
                    # dldr = (log(gsd.gsd_prob(psi , rho+h, k)) - log(
                    #     gsd.gsd_prob(psi, rho, k))) / h
                    #print(dldt,(dldp, dldr))
                    for d in dldt:
                        if np.isnan(d) or np.isinf(d):
                            print(dict(psi=psi, rho=rho, k=k))
                            print(dldt)

                    ...


class FancyMathTestCase(unittest.TestCase):
    def test_prod(self):
        from gsd.ref_prob import 𝚷,ℤ

        x = 𝚷(i for i in ℤ[1,3])
        self.assertEqual(x,6)

class SufficientStatistic(unittest.TestCase):

    def test_sufficient_statistic(self):
        data=[1,2,2,3,4,1,1]
        ss = gsd.sufficient_statistic(data)
        #                                          1, 2  3  4  5
        self.assertTrue(np.allclose(ss,np.asarray([3, 2, 1, 1, 0])))

    def test_sufficient_statistic2(self):
        data=[1,2,3,4,5]
        ss = gsd.sufficient_statistic(data)
        #                                          1, 2  3  4  5
        self.assertTrue(np.allclose(ss,np.asarray([1, 1, 1, 1, 1])))

    def test_sufficient_statistic3(self):
        data=[1,]
        ss = gsd.sufficient_statistic(data)
        #                                          1, 2  3  4  5
        self.assertTrue(np.allclose(ss,np.asarray([1, 0, 0, 0, 0])))

    def test_sufficient_statistic4(self):
        c = [0,0,1,10,13]
        data = sum([int(n) * [i + 1, ] for i, n in enumerate(c)], [])
        ss = gsd.sufficient_statistic(data)
        #                                          1, 2  3  4  5
        self.assertTrue(np.allclose(ss,c))

if __name__ == '__main__':
    unittest.main()
