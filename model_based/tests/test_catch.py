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
        # for _ in range(random.randint(0, 5)):
        action = jax.random.randint(key, shape=(), minval=0, maxval=3)
        obs, env_state, _, _, _ = env.step(key, env_state, action, env_params)

        _, key = jax.random.split(key)
        yield obs, action


def pretty_compare(a, b):
    if (a == b).all():
        return True
    else:
        print()
        print(a.reshape(10, 5))
        print(b.reshape(10, 5))
        return False


def test_model_equivariance():
    model = tm.CatchEquiModel()
    key = jax.random.PRNGKey(10)
    state_gen = mock_catch(1000)
    obs, action = next(state_gen)
    model_params = model.init(key, obs, action)
    apply_fn = jax.jit(model.apply)
    sol = []
    for obs, action in state_gen:
        ball_dist, pad_dist = apply_fn(model_params, obs, action)
        pred = model.dist_to_obs(ball_dist, pad_dist).reshape(10, 5)

        inv_action = tm.catch_action_transform(action)
        inv_obs = catch_transform(obs)
        if (inv_obs == obs).all():
            print("Passing")
            continue

        inv_ball, inv_pad = apply_fn(model_params, inv_obs, inv_action)
        inv_pred = model.dist_to_obs(inv_ball, inv_pad)
        inv_pred = catch_transform(inv_pred).reshape(10, 5)

        # if not (inv_obs == obs).all():
        comparison = pretty_compare(pred, inv_pred)

        if not comparison:
            print()
            print(obs.reshape(10, 5))
        sol.append(comparison)
    print("Fraction of test passed", sum(sol) / len(sol))
    assert all(sol)
