# API

## Reference implementation in python

In order to keep the reference implementation as close to the math as possible we define some utilities with unicode symbols.
E.g.  `𝚷(i for i in ℤ[1,3])` is a valid python code for 

$\prod_{i=1}^{3} i$ 


::: gsd.gsd_prob
    options:
      show_root_heading: true

## JAX functions

Distribution functions implemented in JAX for speed and auto differentiation.

__Currently, we support only GSD with 5 point scale__

::: gsd.log_prob 

---

::: gsd.sample

---

::: gsd.mean

---

::: gsd.variance

---

::: gsd.sufficient_statistic


## Fit

We provide few estimators. The simple one is based on moments. 
A more advanced gradient-based estimator maximum likelihood estimator is 
provided in `gsd.experimental`. We also provide a naive grid search MLE.
Besides the high-level API one can use optimizers form `scipy` or `tensorflow_probability`.   

::: gsd.fit_moments


### Constrained parameter space

:::gsd.fit.log_pmax

:::gsd.fit.allowed_region




### Structures


::: gsd.fit.GSDParams


## Experimental

::: gsd.experimental






