import random

import distrax
import jax
from flax import linen as nn
from jax import numpy as jnp

from g_conv.c2 import C2Conv

# def test_shaping():
#     mod = C2Conv(features=3, kernel_size=(1, 4))
#     dummy_state = jnp.ones((1, 1, 4))
#     params = mod.init(jax.random.PRNGKey(0), dummy_state)
#     out = mod.apply(params, dummy_state)
#     assert out.shape == (1, 2, 3), f"Expected shape (1, 2, 3), got {out.shape}"


class TwoLayer(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = C2Conv(features=self.features, kernel_size=((input.shape[1],)))
        out = layer(input)
        layer_2 = C2Conv(features=self.features, kernel_size=((input.shape[1],)))
        out = layer_2(out)
        layer_3 = C2Conv(features=1, kernel_size=((input.shape[1],)))
        out = layer_3(out)
        return out


def mock_model():
    mod = TwoLayer(features=64)
    dummy_state = jnp.ones((1, 4))
    params = mod.init(jax.random.PRNGKey(0), dummy_state)
    return params, mod.apply


def test_multi_layer():
    params, apply_fn = mock_model()
    dummy_state = jax.random.normal(jax.random.PRNGKey(random.randint(0, 1000)), (1, 4))
    out = apply_fn(params, dummy_state)
    r_dummy_state = -dummy_state
    r_out = apply_fn(params, r_dummy_state)
    assert jnp.allclose(out, -r_out)
    assert out.squeeze().shape == (2,)


def test_equiv():
    """Assert that the output of the model is equivariant to the input transformation"""
    params, apply_fn = mock_model()
    dummy_state = jax.random.normal(jax.random.PRNGKey(random.randint(0, 1000)), (1, 4))
    out = apply_fn(params, dummy_state)
    r_dummy_state = -dummy_state
    r_out = apply_fn(params, r_dummy_state)

    dist = distrax.Categorical(logits=out)
    r_dist = distrax.Categorical(logits=r_out)
    assert jnp.allclose(dist.log_prob(0), r_dist.log_prob(1))
