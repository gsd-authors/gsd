import argparse
import jax.numpy as jnp
from gsd import fit_mle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSD estimator')

    parser.add_argument("response", nargs=5, type=int,
                        metavar=("num1", "num2", "num3", "num4", "num5"),
                        help="List of 5 counts")

    args = parser.parse_args()

    hat,_ = fit_mle(data=jnp.asarray(args.response, dtype=jnp.float32))
    print(hat)
