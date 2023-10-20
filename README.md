# GSD

[**Installation**](#installation)
| [**Documentation**](https://gsd-authors.github.io/gsd)
| [**Cite us**](#citeus)


Reference implementation of generalised score distribution in python

This library provides a reference implementation of gsd probabilities for correctness and efficient implementation of samples and log_probabilities in `jax`. 

### Citations<a id="citeus"></a>

Theoretical derivation of GSD is described in the following paper.

Ćmiel, B., Nawała, J., Janowski, L. , Rusek, K. Generalised score distribution: underdispersed continuation of the beta-binomial distribution. Stat Papers (2023). https://doi.org/10.1007/s00362-023-01398-0

If you decide to apply the concepts presented or base on the provided code, please do refer our related paper.




## Installation<a id="installation"></a>

You can install gsd via `pip`:

```bash
$ pip install ref_gsd
```


**[DOC](https://gsd-authors.github.io/gsd)**

## Development

To develop and modify gsd, you need to install
[`hatch`]([https://hatch.pypa.io](https://hatch.pypa.io)), a tool for Python packaging and
dependency management.

To  enter a virtual environment for testing or debugging, you can run:

```bash
$ hatch shell
```

### Running tests

Gsd uses unitest for testing. To run the tests, use the following command:

```
$ hatch run test 
```

### Standalone estimator

You can quickly estimate GSD parameters from a command line interface

```shell
python3 -m gsd -c 1 2 3 4 5
```

    psi=3.6667 rho=0.6000