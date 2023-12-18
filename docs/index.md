# GSD

Reference implementation of generalised score distribution in python

This library provides a reference implementation of gsd probabilities for correctness and efficient implementation of samples and log_probabilities in `jax`.

### Citations

Theoretical derivation of GSD is described in the following papers.

```
@Article{Cmiel2023,
author={{\'{C}}miel, Bogdan
and Nawa{\l}a, Jakub
and Janowski, Lucjan
and Rusek, Krzysztof},
title={Generalised score distribution: underdispersed continuation of the beta-binomial distribution},
journal={Statistical Papers},
year={2023},
month={Feb},
day={09},
issn={1613-9798},
doi={10.1007/s00362-023-01398-0},
url={https://doi.org/10.1007/s00362-023-01398-0}
}

```

```
@ARTICLE{gsdnawala,
  author={Nawała, Jakub and Janowski, Lucjan and Ćmiel, Bogdan and Rusek, Krzysztof and Pérez, Pablo},
  journal={IEEE Transactions on Multimedia}, 
  title={Generalized Score Distribution: A Two-Parameter Discrete Distribution Accurately Describing Responses From Quality of Experience Subjective Experiments}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TMM.2022.3205444}
  }
```

If you decide to apply the concepts presented or base on the provided code, please do refer our related paper.

## Installation

You can install gsd via `pip`:

```bash
pip install ref_gsd
```

***Note that you install `ref_gsd` but import `gsd` e.g.***

```python
import gsd

gsd.fit_moments([2, 8, 2, 0, 0.])
```

## Development

To develop and modify gsd, you need to install
[`hatch`]([https://hatch.pypa.io](https://hatch.pypa.io)), a tool for Python packaging and
dependency management.

To  enter a virtual environment for testing or debugging, you can run:

```bash
hatch shell
```

### Running tests

Gsd uses unitest for testing. To run the tests, use the following command:

```
hatch run test 
```

### Standalone estimator

You can quickly estimate GSD parameters from a command line interface

```shell
python3 -m gsd -c 1 2 3 4 5
```

    psi=3.6667 rho=0.6000
