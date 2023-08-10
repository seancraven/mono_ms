import random
from typing import Generator, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
from base_rl.models import catch_transform
from base_rl.wrappers import FlattenObservationWrapper
from g_conv.c2 import C2Dense
from gymnax.environments.bsuite import Catch

import model_based.transition_models as tm


def mock_catch(num_states) -> Generator[Tuple[jt.Array, jt.Array], None, None]:
    key = jax.random.key(10)
    env = FlattenObservationWrapper(Catch())
    env_params = env.default_params
    obs, env_state = env.reset(key, env_params)
    action = jax.random.randint(key, shape=(), minval=0, maxval=3)
    for _ in range(num_states):
        for _ in range(random.randint(0, 5)):
            action = jax.random.randint(key, shape=(), minval=0, maxval=3)
            obs, env_state, _, _, _ = env.step(key, env_state, action, env_params)

            _, key = jax.random.split(key)
        yield obs, action


def test_transform():
    key = jax.random.PRNGKey(10)
    state_gen = mock_catch(10)
    model = tm._CatchEquiLogits()
    # model = Embedding(5)
    obs, action = next(state_gen)
    model_params = model.init(key, obs, action)
    for obs, action in state_gen:
        next_state_logits = model.apply(model_params, obs, action)
        inv_action = tm.catch_action_transform(action)
        inv_obs = catch_transform(obs)
        next_state_inv = model.apply(model_params, inv_obs, inv_action)
        if not (next_state_logits == jnp.roll(next_state_inv, 1, axis=-1)).all():
            print(next_state_inv)
            print(next_state_logits)
            assert False


def pretty_compare(a, b):
    if (a == b).all():
        return True
    else:
        print(a.reshape(10, 5))
        print(b.reshape(10, 5))
        return False


def test_proximal_catch_pool():
    state_gen = mock_catch(10)
    for i, (obs, action) in enumerate(state_gen):
        obs_inv = catch_transform(obs)
        stacked_states = jnp.stack([obs, obs_inv], axis=-1)
        if i % 2:
            print()
            target_state = obs
        else:
            target_state = obs_inv
        next_state = tm.catch_proximal_state_pool(stacked_states, target_state)
        assert pretty_compare(next_state, target_state)


def test_model():
    model = tm.SimpleCatchEqui()
    key = jax.random.PRNGKey(10)
    state_gen = mock_catch(5)
    obs, action = next(state_gen)
    model_params = model.init(key, obs, action)
    sol = []
    for obs, action in state_gen:
        ball_dist, pad_dist = model.apply(model_params, obs, action)
        pred = model.dist_to_obs(ball_dist, pad_dist).reshape(10, 5)

        inv_action = tm.catch_action_transform(action)
        inv_obs = catch_transform(obs)

        inv_ball, inv_pad = model.apply(model_params, inv_obs, inv_action)
        inv_pred = model.dist_to_obs(inv_ball, inv_pad)
        inv_pred = catch_transform(inv_pred).reshape(10, 5)

        sol.append(pretty_compare(pred, inv_pred))
    print(sum(sol) / len(sol))
    assert all(sol)


if __name__ == "__main__":
    test_model()
