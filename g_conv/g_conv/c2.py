from typing import Callable, Tuple

import jax.numpy as jnp
import jaxtyping as jt
from flax import linen as nn


class C2Conv(nn.Module):
    features: int
    kernel_size: Tuple[int, ...]
    transform: Callable[[jt.Array], jt.Array] = lambda x: -x

    @nn.compact
    def __call__(self, input):
        layer = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
        )
        return jnp.concatenate([layer(input), layer(self.transform(input))], axis=-1)


class C2Dense(nn.Module):
    features: int
    transform: Callable[[jt.Array], jt.Array] = lambda x: -x
    use_bias: bool = False

    @nn.compact
    def __call__(self, input):
        layer = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
        )

        return jnp.stack(
            [layer(input), layer(self.transform(input))],
            axis=-1,
        ).squeeze()


class C2DenseBinary(nn.Module):
    """Special layer that maps 0,1 to 1, -1 and then passes through equivariant layer"""

    features: int

    @nn.compact
    def __call__(self, input):
        map_input = 2 * input - 1
        return C2Dense(features=self.features)(map_input)


if __name__ == "__main__":
    pass
