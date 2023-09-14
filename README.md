# gsd
Reference implementation of generalised score distribution in python

This library provides a reference implementation of gsd probabilities for correctness and efficient implementation of samples and log_probabilities in `jax`. 

### Citations

Theoretical derivation of GSD is described in the following paper.

Ćmiel, B., Nawała, J., Janowski, L. et al. Generalised score distribution: underdispersed continuation of the beta-binomial distribution. Stat Papers (2023). https://doi.org/10.1007/s00362-023-01398-0

If you decide to apply the concepts presented or base on the provided code, please do refer our related paper.

### Fancy math

In order to keep the reference implementation as close to the math as possible we define some utilities with unicode symbols.
E.g.  `𝚷(i for i in ℤ[1,3])` is a valid python code for $$\prod_{i=1}^{3} i$$




## Installation

You can install gsd via `pip`:

```bash
$ pip install ref_gsd
```



## Development

To develop and modify gsd, you need to install
[`hatch`]([https://python-poetry.org/](https://hatch.pypa.io)), a tool for Python packaging and
dependency management.

To  enter a virtual environment for testing or debugging, you can run:

```bash
$ hatch shell
```

### Running tests

Gsd uses [Pytest](https://pytest.org/) for testing. To run the tests, use the following command:

```
$ hatch run test 
```
