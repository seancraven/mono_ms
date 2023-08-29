import distrax
import random
import jax
from flax import linen as nn
from jax import numpy as jnp

from g_conv.c2 import C2Conv, C2Dense
from model_based.transition_models import hidden_transform, proximal_state_pool

# def test_shaping():
#     mod = C2Conv(features=3, kernel_size=(1, 4))
#     dummy_state = jnp.ones((1, 1, 4))
#     params = mod.init(jax.random.PRNGKey(0), dummy_state)
#     out = mod.apply(params, dummy_state)
#     assert out.shape == (1, 2, 3), f"Expected shape (1, 2, 3), got {out.shape}"


class ConvEqui(nn.Module):
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


def mock_model():
    mod = ConvEqui(features=64)
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


class Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(self.features, use_bias=False)
        out = nn.tanh(layer(input))
        layer_2 = nn.Dense(self.features, use_bias=False)
        out = nn.tanh(layer_2(out))
        layer_3 = C2Dense(features=1)
        out = layer_3(out)
        return distrax.Categorical(logits=out)


def test_dense_equiv():
    layer = C2Dense(features=3, use_bias=False)

    params = layer.init(jax.random.PRNGKey(0), jnp.ones((2,)))
    apply_fn = jax.vmap(layer.apply, in_axes=(None, 0))
    out = apply_fn(params, jnp.ones((100, 2)))
    inv = apply_fn(params, -jnp.ones((100, 2)))

    assert (out == jnp.roll(inv, 1, axis=-1)).all()


class DenseDeep(nn.Module):
    h_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1)

        out = C2Dense(
            self.h_dim,
        )(x)
        out = nn.relu(out.reshape(-1))
        print(out.shape)
        out = C2Dense(self.h_dim, use_bias=True, transform=hidden_transform)(out)
        out = nn.relu(out.reshape(-1))
        print(out.shape)
        out = C2Dense(1, use_bias=True, transform=hidden_transform)(out)
        out = convert_group_action(out)
        return proximal_state_pool(x, out)


def convert_group_action(stacked_logits):
    idn_logits = stacked_logits.at[..., 0].get()
    inv_logits = stacked_logits.at[..., 1].get()
    inv_logits = -(inv_logits)
    return jnp.stack([idn_logits, inv_logits], axis=-1)


def test_dense_equiv_deep():
    model = DenseDeep(h_dim=64)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((4)))
    input = jax.random.normal(jax.random.PRNGKey(4), (100, 4))
    v_app = jax.vmap(model.apply, in_axes=(None, 0))
    out = v_app(params, input)
    r_out = v_app(params, -input)
    assert (out == -r_out).all()


def test_dense():
    model = Dense(features=64)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((4)))
    input = jax.random.normal(jax.random.PRNGKey(4), (4,))
    out = model.apply(params, input)
    r_out = model.apply(params, -input)
    assert out.log_prob(0) == r_out.log_prob(1)
