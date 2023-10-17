import argparse
import jax.numpy as jnp
from gsd import fit_mle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSD estimator')
    parser.add_argument("-c", nargs=5, type=int,help="List of 5 counts", required=True)
    args = parser.parse_args()

    hat,_ = fit_mle(data=jnp.asarray(args.o, dtype=jnp.float32))
    print(f'psi={hat.psi:.4f} rho={hat.rho:.4f}')
