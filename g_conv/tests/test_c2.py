import random

import distrax
import jax
from base_rl.models import EquivariantActorCritic
from flax import linen as nn
from jax import numpy as jnp

from g_conv.c2 import ActionEquiv, C2Conv, C2Dense, C2DenseLift

# def test_shaping():
#     mod = C2Conv(features=3, kernel_size=(1, 4))
#     dummy_state = jnp.ones((1, 1, 4))
#     params = mod.init(jax.random.PRNGKey(0), dummy_state)
#     out = mod.apply(params, dummy_state)
#     assert out.shape == (1, 2, 3), f"Expected shape (1, 2, 3), got {out.shape}"


def test_action_equiv():
    layer = ActionEquiv(features=3)
    params = layer.init(jax.random.PRNGKey(0), jnp.ones((1,)))
    out = layer.apply(params, jnp.ones((1,)))
    inv = layer.apply(params, jnp.zeros((1,)))
    print(out, inv)
    assert jnp.allclose(out, -inv)


class TwoLayer(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = C2Conv(features=self.features, kernel_size=((1,)))
        out = nn.tanh(layer(input))
        layer_2 = C2Conv(features=self.features, kernel_size=((1,)))
        out = layer_2(out)
        layer_3 = C2Conv(features=1, kernel_size=((1,)))
        out = layer_3(out)
        return out


class Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(self.features, use_bias=False)
        out = nn.tanh(layer(input))
        layer_2 = nn.Dense(self.features, use_bias=False)
        out = nn.tanh(layer_2(out))
        layer_3 = C2DenseLift(features=1)
        out = layer_3(out)
        return distrax.Categorical(logits=out)


def test_dense():
    model = Dense(features=64)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((4)))
    input = jax.random.normal(jax.random.PRNGKey(4), (4,))
    out = model.apply(params, input)
    r_out = model.apply(params, -input)
    assert out.log_prob(0) == r_out.log_prob(1)


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


def test_params():
    model = C2Conv(features=64, kernel_size=(1,))
    dummy_state = jnp.ones((1, 4))
    params = model.init(jax.random.PRNGKey(0), dummy_state)
    single_lyaer = nn.Conv(features=64, kernel_size=(1,))
    single_params = single_lyaer.init(jax.random.PRNGKey(0), dummy_state)
    size_c2 = sum(x.size for x in jax.tree_leaves(params))
    size_conv = sum(x.size for x in jax.tree_leaves(single_params))
    assert size_c2 == size_conv
