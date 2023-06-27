from __future__ import annotations

from typing import Protocol, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, PRNGKeyArray


class Symmetrizer(nn.Module):
    input_dim: int
    output_dim: int
    Group: Group

    @nn.compact
    def __call__(self, input):
        # TODO: Implement the forward pass I think I have dont the hard part.
        pass

    def _find_basis(self, key: PRNGKeyArray, samples: int = 100) -> List[Array]:
        matricies = jax.random.normal(
            jax.random.PRNGKey(0), (samples, self.input_dim, self.output_dim)
        )
        symmetrized_mats = _symmetrize(self.Group, matricies).reshape(samples, -1)
        _, _, V = jnp.linalg.svd(symmetrized_mats)
        r = jnp.linalg.matrix_rank(V)
        return V[:r]


def _symmetrize(group: Group, matricies: Array):
    symmetrized_mats = jnp.zeros_like(matricies)
    in_dim = matricies.shape[2]
    out_dim = matricies.shape[1]
    rep_in = group.get_representation(in_dim)
    rep_out = group.get_representation(out_dim)
    for g_in, g_out in zip(rep_in, rep_out):
        symmetrized_mats += jax.vmap(lambda x: _sym(g_in, x, g_out), in_axes=0)(
            matricies
        )

    return symmetrized_mats


def _sym(in_: Array, out: Array, mat: Array) -> Array:
    out_inv = jnp.linalg.inv(out)
    return jnp.einsum("ij,jk,kl->il", out_inv, mat, in_)


class Group(Protocol):
    size: int

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        "Returns an interable of the representation in the given dimension"
        ...


class C2Group(Group):
    size = 2

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        return jnp.eye(dim), -jnp.eye(dim)
