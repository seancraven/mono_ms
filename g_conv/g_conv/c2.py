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


if __name__ == "__main__":
    pass