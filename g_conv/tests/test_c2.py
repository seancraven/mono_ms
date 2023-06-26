import distrax
import jax
from flax import linen as nn
from jax import numpy as jnp

from c2 import C2Conv

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


def test_multi_layer():
    mod = TwoLayer(features=64)
    dummy_state = jnp.ones((1, 1, 4))
    params = mod.init(jax.random.PRNGKey(0), dummy_state)
    out = mod.apply(params, dummy_state)
    r_dummy_state = -dummy_state
    r_out = mod.apply(params, r_dummy_state)
    print(out)
    print(r_out)
    assert jnp.allclose(out, -mod.apply(params, r_dummy_state))
    assert out.squeeze().shape == (2,)
    assert (nn.softmax(out)[0] == nn.softmax(-r_out)[0]).all()
