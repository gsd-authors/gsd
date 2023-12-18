from jax import config
config.update("jax_enable_x64", True)

import unittest  # noqa: E402

import jax
import jax.numpy as jnp
import numpy as np

import gsd
from gsd.experimental import fit_mle_grid, bootstrap
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


class BootstrapTestCase(unittest.TestCase):
    def test_sample_fit(self):
        k = jax.random.key(12)
        th = GSDParams(psi=4.2, rho=.92)
        th = jax.tree_util.tree_map(jnp.asarray, th)
        s = gsd.sample(th.psi, th.rho, (100000,),k)
        data = gsd.sufficient_statistic(s)
        num = GSDParams(512, 128)
        grid = GridEstimator.make(num)
        hat = grid(data)
        self.assertAlmostEqual(hat.psi, th.psi, 2)
        self.assertAlmostEqual(hat.rho, th.rho, 2)
        ...

    def test_g_test(self):
        # https://github.com/Qub3k/gsd-acm-mm/blob/master/Data_Analysis/G-test_results/G_test_on_real_data_chunk000_of_872.csv
        data = jnp.asarray([0,0,1,10,13.])
        num = GSDParams(512, 128)
        grid = GridEstimator.make(num)


        hat = grid(data)
        self.assertTrue(np.allclose(hat.psi, 4.5, 0.001))
        self.assertTrue(np.allclose(hat.rho, 0.935, 0.01))

        p = bootstrap.prob(hat)
        # 0.09459716927725387
        t = bootstrap.t_statistic(data,p)
        self.assertAlmostEqual(t,0.09459716927725387,2)

        # 0.4957
        pv = bootstrap.pp_plot_data(data,lambda x: grid(x) ,jax.random.key(44),9999)

        self.assertAlmostEqual(pv,0.4957,1)


        ...