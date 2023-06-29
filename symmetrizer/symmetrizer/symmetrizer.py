from __future__ import annotations

from typing import Protocol, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal, zeros_init
from jaxtyping import Array, PRNGKeyArray, Float


class Symmetrizer(nn.Module):
    """A linear Symmetrizer Layer: https://arxiv.org/abs/2006.16908

    Args:
        key: A PRNGKey used to initialize the layer.
        in_dim: The input dimension of the layer.
        out_dim: The output dimension of the layer.
        group: The group to symmetrize under.
        samples: The number of samples to use to find the basis.
        bias: Whether to include a bias term in the layer.
    """

    key: PRNGKeyArray
    in_dim: int
    out_dim: int
    group: Group
    samples: int = 100
    bias: bool = True

    def setup(self):
        if self.bias:
            _, bias_key = jax.random.split(self.key)
            self.bias_basis = _find_basis(
                bias_key, self.group, 1, self.out_dim, samples=self.samples
            )
            bias_space_rank = self.bias_basis.shape[0]
            self.bias_coef = self.param(
                "bias basis coefficients", zeros_init(), (bias_space_rank, 1)
            )

        self.basis = _find_basis(
            self.key, self.group, self.in_dim, self.out_dim, samples=self.samples
        )
        sub_space_rank = self.basis.shape[0]
        self.basis_coef = self.param(
            "basis coefficients", lecun_normal(), (sub_space_rank, 1)
        )

    def __call__(self, input: Array) -> Array:
        symmetrizer_mat = jnp.einsum(
            "ijk,il->jkl", self.basis, self.basis_coef
        ).reshape(self.out_dim, self.in_dim)

        out = jnp.einsum("ij,j->i", symmetrizer_mat, input)
        if self.bias:
            bias = jnp.einsum("ijk,il->jkl", self.bias_basis, self.bias_coef).squeeze()
            return out + bias
        else:
            return out


def _find_basis(
    key: PRNGKeyArray, group: Group, in_dim: int, out_dim: int, samples: int = 100
) -> Float[Array, "rank out in"]:
    """Finds an orthogonal basis for the subspace of matricies that are
    the solution to W = _symmetrize(group, W)

    """
    matricies = jax.random.normal(key, (samples, out_dim, in_dim))
    symmetrized_mats = _symmetrize(group, matricies).reshape(samples, -1)
    _, _, V = jnp.linalg.svd(symmetrized_mats)
    r = jnp.linalg.matrix_rank(V)
    return V[:r].reshape(r, out_dim, in_dim)


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
        assert g_in.shape == (in_dim, in_dim), f"{g_in.shape} {in_dim}"
        assert g_out.shape == (out_dim, out_dim), g_out.shape
        symmetrized_mats += jax.vmap(lambda x: _sym(g_in, g_out, x), in_axes=0)(
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
        return (jnp.eye(dim), -jnp.eye(dim))


class C2PermGroup(Group):
    size = 2

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        if dim == 2:
            return (jnp.eye(2), jnp.array([[0, 1], [1, 0]]))
        else:
            return (jnp.eye(dim), -jnp.eye(dim))
