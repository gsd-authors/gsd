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
::: gsd.sample
::: gsd.mean
::: gsd.variance
::: gsd.sufficient_statistic


## Fit

We provide two estimators. 
The simple one based on moments and the maximum likelihood estimator.

::: gsd.fit_mle

::: gsd.fit_moments

## Structures


::: gsd.fit.GSDParams

::: gsd.fit.OptState






