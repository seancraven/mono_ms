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
        assert input.shape[1] == 2
        layer = nn.Dense(
            features=self.features,
            kernel_init=nn.initializers.he_normal(),
        )
        return jnp.stack([layer(input[:, 0]), layer(-input[:, 1])], axis=1)


class C2DenseLift(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(
            features=self.features,
            kernel_init=nn.initializers.he_normal(),
        )
        return jnp.stack([layer(input), layer(-input)], axis=1)


class C2DenseLiftDiscrete(nn.Module):
    features: int

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(
            features=self.features,
        )
        return jnp.stack([layer(input), layer(1 - input)], axis=1).squeeze()


if __name__ == "__main__":
    pass
