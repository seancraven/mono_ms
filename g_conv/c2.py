from typing import Callable, Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import DotGeneralT, Dtype, PrecisionLike, PRNGKey, Shape
from jax import lax
from jaxtyping import Array

InitFn = Callable[[PRNGKey, Shape, Dtype], Array]


class C2Conv(nn.Module):

    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: InitFn = initializers.lecun_normal()  # type: ignore
    bias_init: InitFn = initializers.zeros_init()  # type: ingore
    dot_general: DotGeneralT = lax.dot_general

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """applies a linear transformation to the inputs along the last dimension.

        args:
          inputs: the nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        y_prime = self.dot_general(
            -inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y_prime += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return jnp.stack([y, y_prime], axis=-1)


if __name__ == "__main__":
    pass
