from random import randint

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import flax.linen as nn
from distrax import Categorical

from symmetrizer.symmetrizer import (
    C2Group,
    C2PermGroup,
    Symmetrizer,
    _find_basis,
    _sym,
    _symmetrize,
)


def test_find_basis():
    group = C2Group()
    key = jax.random.PRNGKey(randint(0, 1000))
    basis = _find_basis(key, group, 2, 2, samples=1000)
    symed_basis = _symmetrize(group, basis)
    assert jax.numpy.allclose(basis, symed_basis)


def test_sym():
    in_dim = randint(1, 10)
    out_dim = randint(1, 10)
    reflction_in = -jnp.eye(in_dim)
    reflction_out = -jnp.eye(out_dim)
    ref_out_inv = jnp.linalg.inv(reflction_out)
    id_in = jnp.eye(in_dim)
    id_out = jnp.eye(out_dim)
    id_out_inv = jnp.linalg.inv(id_out)
    mat = jax.random.normal(jax.random.PRNGKey(0), (out_dim, in_dim))
    assert (
        _sym(reflction_in, reflction_out, mat) == ref_out_inv @ mat @ reflction_in
    ).all()
    assert (_sym(id_in, id_out, mat) == id_out_inv @ mat @ id_in).all()


def test_symmetrize():
    group = C2Group()
    in_dim = randint(1, 10)
    out_dim = randint(1, 10)
    in_rep = group.get_representation(in_dim)
    out_rep = group.get_representation(out_dim)
    inv_out_rep = [jnp.linalg.inv(g) for g in out_rep]
    mat = jax.random.normal(jax.random.PRNGKey(0), (out_dim, in_dim))
    symed_mat = _symmetrize(group, mat[None, ...])
    maual_sym_mat = sum(k @ mat @ g for k, g in zip(inv_out_rep, in_rep)) / group.size
    assert jnp.allclose(symed_mat, maual_sym_mat)


def test_linear():
    in_dim = randint(1, 100)
    out_dim = randint(1, 100)
    key = jax.random.PRNGKey(randint(0, 1000))
    layer = Symmetrizer(
        key,
        in_dim,
        out_dim,
        C2Group(),
        bias=False,
    )
    _, key = jax.random.split(key)
    x = 100 * jax.random.normal(key, (100, in_dim))
    layer_params = layer.init(key, x[0])
    vec_apply = jax.vmap(layer.apply, in_axes=(None, 0))
    y = vec_apply(layer_params, x)
    y_bar = vec_apply(layer_params, -x)
    assert jnp.allclose(y, -y_bar)
    assert not jnp.allclose(y, 0)


def test_bias():
    in_dim = randint(1, 100)
    out_dim = randint(1, 100)
    key = jax.random.PRNGKey(randint(0, 1000))
    layer = Symmetrizer(
        key,
        in_dim,
        out_dim,
        C2Group(),
        bias=True,
        samples=1000,
    )
    _, key = jax.random.split(key)
    x = jax.random.normal(key, (100, in_dim))
    layer_params = layer.init(key, x[0])
    vec_apply = jax.vmap(layer.apply, in_axes=(None, 0))
    y = vec_apply(layer_params, x)
    y_bar = vec_apply(layer_params, -x)
    assert jnp.allclose(y, -y_bar)
    assert not jnp.allclose(y, 0)


class EquivariantMLP(nn.Module):
    key: PRNGKeyArray
    in_dim: int = 4
    internal_dim: int = 64
    out_dim: int = 1

    @nn.compact
    def __call__(self, in_):
        _, key_1, key_2, key_3 = jax.random.split(self.key, 4)
        out = Symmetrizer(
            key_1,
            self.in_dim,
            self.internal_dim,
            C2Group(),
            bias=True,
            samples=1000,
        )(in_)
        out = nn.tanh(out)
        out = Symmetrizer(
            key_2,
            self.internal_dim,
            self.internal_dim,
            C2Group(),
            bias=False,
            samples=1000,
        )(out)
        out = nn.tanh(out)
        out = Symmetrizer(
            key_3,
            self.internal_dim,
            self.out_dim,
            C2PermGroup(),
            bias=False,
            samples=1000,
        )(out)
        return Categorical(logits=out)


def test_mlp():
    in_dim = 4
    out_dim = 2
    key = jax.random.PRNGKey(randint(0, 1000))

    x = 100 * jax.random.normal(key, (5, in_dim))
    _, key = jax.random.split(key)
    model = EquivariantMLP(key, in_dim, out_dim=out_dim)
    _, key = jax.random.split(key)
    params = model.init(key, x[0])
    apply = jax.vmap(model.apply, in_axes=(None, 0))
    p_0 = apply(params, x).log_prob(1)
    inverted_p_1 = apply(params, -x).log_prob(0)
    print(p_0, inverted_p_1)
    assert jnp.allclose(p_0, inverted_p_1)
