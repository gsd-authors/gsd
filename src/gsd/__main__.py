import gsd
import jax
import jax.numpy as jnp


if __name__ == '__main__':
    gsd.log_prob(1., 0.5, 2)
    m=gsd.mean(3.,0.7)
    v = gsd.variance(3.,0.7)
    k = jax.random.key(43)
    s = gsd.sample(3.,0.7,(24,),k)

    jnp.mean(s), jnp.var(s)

    #jax.vmap(gsd.log_prob, in_axes=(None,None,0))(3.,0.7,s)




    print('test')