from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from base_rl.models import ACSequential
from distrax import Categorical
from flax import linen as nn
from flax.linen.initializers import lecun_normal, zeros_init
from jaxtyping import Array, Float, PRNGKeyArray


class Group(Protocol):
    """A fininte group that has a representation in a given dimension"""

    size: int

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        "Returns an interable of the representation in the given dimension"
        ...


class C2Group(Group):
    """Reflection group sets x = -x for the reflection operation"""

    size = 2

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        return (jnp.eye(dim), -jnp.eye(dim))


class C2PermGroup(Group):
    """Reflection group that sets x = -x for the reflection operation
    for all dimentions but 2. Where it permutes the elements"""

    size = 2

    def get_representation(self, dim: int) -> Tuple[Array, Array]:
        if dim == 2:
            return (jnp.eye(2), jnp.array([[0, 1], [1, 0]]))
        else:
            return (jnp.eye(dim), -jnp.eye(dim))


class SymmetrizerDense(nn.Module):
    """A Jitable symmetrizer Layer to be called from inside a factory

    Args:
        basis: The basis for the subspace of matricies,
        which solve:
            W = _symmetrize(group, W)
        bias_basis: Optional, the basis for the subspace of matricies,
        which solve:
            _symmetrize(group, b)
    """

    basis: Array
    bias_basis: Optional[Array] = None

    def setup(self):
        sub_space_rank = self.basis.shape[0]
        self.basis_coef = self.param(
            "basis coefficients", lecun_normal(), (sub_space_rank, 1)
        )
        if self.bias_basis is not None:
            bias_space_rank = self.bias_basis.shape[0]
            self.bias_coef = self.param(
                "bias basis coefficients", zeros_init(), (bias_space_rank, 1)
            )

    def __call__(self, input: Array) -> Array:
        symmetrizer_mat = jnp.einsum(
            "ijk,il->jkl", self.basis, self.basis_coef
        ).squeeze()

        out = jnp.einsum("ij,...j->...i", symmetrizer_mat, input)
        if self.bias_basis is not None:
            bias = jnp.einsum("ijk,il->jkl", self.bias_basis, self.bias_coef).squeeze()
            return out + bias
        else:
            return out


class Sequential(nn.Module):
    layer_list: Tuple[nn.Module, ...]

    @nn.compact
    def __call__(self, input: Array) -> Categorical:
        out = input
        for layer in self.layer_list:
            out = layer(out)
        return Categorical(logits=out)


def ac_symmmetrizer_factory(
    key: PRNGKeyArray, group: Group, layer_list: List[int], bias_list: List[bool]
) -> ACSequential:
    """A factory for creating a symmetrized actor-critic network

    Args:
        key: A PRNGKeyArray
        group: A Group
        layer_list: A list of layer sizes
        bias_list: A list of booleans indicating whether the layer should have a bias
    """
    if len(layer_list) != len(bias_list):
        raise ValueError("layer_list and bias_list must have the same length")

    actor_layers = []
    critic_layers = []
    for in_dim, out_dim, bias in zip(layer_list[:-1], layer_list[1:], bias_list):
        basis = _find_basis(key, group, in_dim, out_dim)
        if bias:
            bias_basis = _find_basis(key, group, 1, out_dim)
            actor_layers.append(SymmetrizerDense(basis, bias_basis))
            actor_layers.append(nn.tanh)
        else:
            actor_layers.append(SymmetrizerDense(basis))
            actor_layers.append(nn.tanh)
        # Rmove the final nonlinearity
        critic_layers.append(nn.Dense(out_dim, use_bias=bias))
        critic_layers.append(nn.relu)
    actor_layers.pop(-1)
    critic_layers.pop(-1)
    critic_layers.pop(-1)
    critic_layers.append(nn.Dense(1, use_bias=True))
    return ACSequential(tuple(actor_layers), tuple(critic_layers))


def symmmetrizer_factory(
    key: PRNGKeyArray, group: Group, layer_list: List[int], bias_list: List[bool]
):
    """A factory for creating a symmetrized sequentail dense network

    Args:
        key: A PRNGKeyArray
        group: A Group
        layer_list: A list of layer sizes
        bias_list: A list of booleans indicating whether the layer should have a bias
    """

    if len(layer_list) != len(bias_list):
        raise ValueError("layer_list and bias_list must have the same length")

    layers = []
    for in_dim, out_dim, bias in zip(layer_list[:-1], layer_list[1:], bias_list):
        basis = _find_basis(key, group, in_dim, out_dim)
        if bias:
            bias_basis = _find_basis(key, group, 1, out_dim)
            layers.append(SymmetrizerDense(basis, bias_basis))
            layers.append(nn.tanh)
        else:
            layers.append(SymmetrizerDense(basis))
            layers.append(nn.tanh)
    # Rmove the final nonlinearity
    layers.pop(-1)
    return Sequential(tuple(layers))


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
