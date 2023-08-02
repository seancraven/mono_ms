import jax
import jax.numpy as jnp
import numpy as np
from g_conv.c2 import C2Dense
from gymnax.environments.bsuite import Catch

from base_rl.models import (EquivariantActorCritic,
                            EquivariantCatchActorCritic, catch_transform)
from base_rl.wrappers import FlattenObservationWrapper


def test_equivaraint_ac():
    rng = jax.random.PRNGKey(0)
    model = EquivariantActorCritic(10)
    params = model.init(rng, jnp.ones((4,)))

    _, rng = jax.random.split(rng)
    state = jax.random.normal(rng, (10, 4))

    def dist_prob(x, i):
        pi, v = model.apply(params, x)
        return pi.log_prob(i)

    get_probs = jax.vmap(dist_prob, in_axes=(0, None))
    assert jnp.allclose(get_probs(state, 0), get_probs(-state, 1))


@jax.jit
def find_ball_loc(single_obs):
    single_obs = single_obs.reshape((10, 5))
    sol = (0, 0)
    for i in reversed(range(9)):
        for j in range(5):
            ret = jax.lax.cond(single_obs[i, j] == 1, lambda: (i, j), lambda: (0, 0))
            sol = jax.tree_map(lambda x, y: x + y, sol, ret)
    return sol


def mock_catch_obs():
    num_envs = 10
    env = Catch()
    env_params = env.default_params
    env = FlattenObservationWrapper(env)
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, num_envs)
    env_rst_fn = jax.vmap(env.reset, in_axes=(0, None))
    obsv, env_state = env_rst_fn(rng, env_params)
    env_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    obsv_new, env_state, rw, done, info = env_step_fn(
        rng, env_state, jnp.zeros(num_envs, dtype=jnp.int32), env_params
    )
    obsv_new, env_state, rw, done, info = env_step_fn(
        rng, env_state, jnp.zeros(num_envs, dtype=jnp.int32), env_params
    )

    obsv = jnp.concatenate([obsv, obsv_new], axis=0)

    return obsv


def test_c2Densetf():
    obsv = mock_catch_obs()

    i_obsv = catch_transform(obsv).reshape((-1, 10, 5))
    obsv = obsv.reshape((-1, 10, 5))
    ball_loc = jax.vmap(find_ball_loc)(obsv)
    i_ball_loc = jax.vmap(find_ball_loc)(i_obsv)
    assert (ball_loc[0] == i_ball_loc[0]).all()
    assert (ball_loc[1] == 4 - i_ball_loc[1]).all()


def test_catch_equiv():
    """Tests that the C2Dense layer with catch transform is equivariant."""
    layer = C2Dense(10, catch_transform)
    rng = jax.random.PRNGKey(0)
    catch_obs = mock_catch_obs()
    params = layer.init(rng, catch_obs[0])
    apply_fn = jax.vmap(layer.apply, in_axes=(None, 0))
    out = apply_fn(params, catch_obs)
    out_trans = apply_fn(params, catch_transform(catch_obs))
    assert (out[..., 0] == out_trans[..., 1]).all()


def test_catch_ac():
    model = EquivariantCatchActorCritic(3)
    rng = jax.random.PRNGKey(42)
    catch_obs = mock_catch_obs()
    params = model.init(rng, catch_obs[0])
    catch_obs = catch_obs[0]
    # apply_fn = jax.vmap(model.apply, in_axes=(None, 0))
    pi, _ = model.apply(params, catch_obs)
    pi_i, _ = model.apply(params, catch_transform(catch_obs))
    print(pi.logits)
    print(pi_i.logits)
    assert jnp.isclose(pi.log_prob(1), pi_i.log_prob(1))
    assert jnp.isclose(pi.log_prob(0), pi_i.log_prob(2))
