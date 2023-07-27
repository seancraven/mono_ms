from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class C2Conv(nn.Module):
    features: int
    kernel_size: Tuple[int, ...]

    @nn.compact
    def __call__(self, input):
        layer = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        return jnp.concatenate([layer(input), layer(-input)], axis=-1)


class C2Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(
            features=self.features,
            use_bias=False,
        )

        return jnp.stack([layer(input), layer(-input)], axis=-1).squeeze()


class C2DenseDiscrete(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(
            features=self.features,
            use_bias=False,  # Affine Transformation is Equivariant to Action inversion
        )
        return jnp.stack([layer(input), layer(1 - input)], axis=-1).squeeze()


class ActionEquiv(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(
            features=self.features,
            use_bias=False,
        )
        return layer(2 * input - 1)


if __name__ == "__main__":
    pass
