import random

import jax
from flax import linen as nn
from jax import numpy as jnp

from c2 import C2Conv


def test_equivariance():
    seed = random.randint(0, 1000)
    key = jax.random.PRNGKey(seed)
    model = C2Conv(features=3)
    in_ = jnp.ones((1, 2, 3))
    model_params = model.init(key, in_)
    output = model.apply(model_params, in_)

    transformed_input = -in_
    transformed_output = model.apply(model_params, transformed_input)
    assert jnp.allclose(transformed_output, -output)


def test_reshape():
    seed = random.randint(0, 1000)
    key = jax.random.PRNGKey(seed)
    model = C2Conv(features=6)
    in_ = jnp.ones((1, 40))
    model_params = model.init(key, in_)
    output = model.apply(model_params, in_)
    output = output.reshape((1, -1, 2))

    transformed_input = -in_
    transformed_output = model.apply(model_params, transformed_input)
    transformed_output = transformed_output.reshape((1, -1, 2))
    assert jnp.allclose(transformed_output, -output)


class Mod(nn.Module):
    internal_dim: int = 4

    @nn.compact
    def __call__(self, input):
        batch_shape = input.shape[0]
        x = C2Conv(features=self.internal_dim)(input)
        x = nn.relu(x)
        x = C2Conv(features=self.internal_dim)(x)
        x = nn.relu(x)
        x = C2Conv(features=1)(x)
        return x


def test_network():
    """Mock network to learn a policy for cartpole."""
    seed = random.randint(0, 1000)
    key = jax.random.PRNGKey(seed)
    mod = Mod()
    in_ = jnp.ones((1, 4))
    tranformed_input = -in_
    model_params = mod.init(key, in_)
    output = mod.apply(model_params, in_)
    transformed_output = mod.apply(model_params, tranformed_input)
    print(transformed_output)
    print("----")
    print(output)
    assert False
