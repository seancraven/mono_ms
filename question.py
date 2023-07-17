import jax
import jax.numpy as jnp


@jax.jit
def names(state, z):
    zs = [z]
    ys = []
    for _ in range(10):
        state, y, z = fn(state, zs)
        zs.append(z)
        ys.append(y)
    return jax.numpy.stack(zs), jax.numpy.stack(ys)


def fn(state, zs):
    z = zs[-1]
    return state, z, z


names(jnp.zeros(1), jnp.ones(1))
