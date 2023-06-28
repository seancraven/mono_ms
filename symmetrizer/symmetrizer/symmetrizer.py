from __future__ import annotations

from typing import Protocol, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal
from jaxtyping import Array, PRNGKeyArray


class Symmetrizer(nn.Module):
    """A linear Symmetrizer Layer: https://arxiv.org/abs/2006.16908"""

    def setup(
        self,
        key,
        in_dim: int,
        out_dim: int,
        group: Group,
        samples: int = 100,
        bias=True,
    ):
        self.in_dim = in_dim
        if bias:
            self.in_dim += 1
        self.out_dim = out_dim
        self.group = Group
        self.basis = _find_basis(key, group, in_dim, out_dim, samples=samples)
        subspace_rank = self.basis.shape[0]
        self.basis_coef = self.param("params", lecun_normal(), (subspace_rank,))

    def __call__(self, input: Array) -> Array:
        return jnp.einsum("ijk,i->jk", self.basis, self.basis_coef) @ input


def _find_basis(
    key: PRNGKeyArray, group: Group, in_dim: int, out_dim: int, samples: int = 100
) -> Array:
    """Finds an orthogonal basis for the subspace of matricies that are
    the solution to W = _symmetrize(group, W)"""
    matricies = jax.random.normal(key, (samples, in_dim, out_dim))
    symmetrized_mats = _symmetrize(group, matricies).reshape(samples, -1)
    _, _, V = jnp.linalg.svd(symmetrized_mats)
    r = jnp.linalg.matrix_rank(V)
    return V[:r].reshape(r, in_dim, out_dim)


def _symmetrize(group: Group, matricies: Array) -> Array:
    """Symmetrizes a set of matricies under the action of a group.

    Maths:
    For some Group element g, and some matrix M dim (n, m) L_g is the action on n
    and K_g is the action on m.

    _symmetrize(G, M) = 1/ |G| sum_{g in G} K_g^-1 @ M @ L_g

    """
    symmetrized_mats = jnp.zeros_like(matricies)
    in_dim = matricies.shape[2]
    out_dim = matricies.shape[1]
    rep_in = group.get_representation(in_dim)
    rep_out = group.get_representation(out_dim)
    for g_in, g_out in zip(rep_in, rep_out):
        symmetrized_mats += jax.vmap(lambda x: _sym(g_in, x, g_out), in_axes=0)(
            matricies
        )

    return symmetrized_mats / group.size


def _sym(in_: Array, out: Array, mat: Array) -> Array:
    """Applies the Inverse left action of the group action on the output space
    on the matrix mat, then the left action on the input space


    Maths:
    For some Group element g, and some matrix M dim (n, m) L is the action on n
    and K is the action on m.


    Args:
        in_: The left action of the group element on the input space.
        out: The inverse left action of the group element on the output space.
        mat: The matrix to be symmetrized.

    Returns:
        k^-1 @ M @ L
    """

    out_inv = jnp.linalg.inv(out)
    return jnp.einsum("ij,jk,kl->il", out_inv, mat, in_)


class Group(Protocol):
    """A fininte group that has a representation in a given dimension"""

    size: int

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        "Returns an interable of the representation in the given dimension"
        ...


class C2Group(Group):
    size = 2

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        return jnp.eye(dim), -jnp.eye(dim)
