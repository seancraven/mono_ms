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
        print("both predictions")
        print(a.reshape(10, 5))
        print(b.reshape(10, 5))
        return False


def test_model_equivariance_exclude_type1():
    """Fails when model predictions are equivariant to
    transformations on the input. This test excludes input states
    that are along the center line and so
    state @ group_action = state.
    """
    model = tm.CatchEquiModel_()
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
            # Exclude states that the ball and paddle are
            # in the middle column, 2, (zero index)
            continue

        inv_ball, inv_pad = apply_fn(model_params, inv_obs, inv_action)
        inv_pred = model.dist_to_obs(inv_ball, inv_pad).reshape(10, 5)
        transform_pred = catch_transform(pred).reshape(10, 5)

        comparison = pretty_compare(transform_pred, inv_pred)

        if not comparison:
            # uncomment lines 35, 37 to see when goes wrong
            # print()
            print("origional state")
            print(obs.reshape(10, 5))
        sol.append(comparison)
    print("Fraction of Predictions that are equivarant", sum(sol) / len(sol))
    assert all(sol), "Not all next state predictions are equivariant"


def test_model_equivariance():
    """Test that when the input state and input action are transformed,
    the output state is transformed in the same manner.
    checks for this property
    group_action @ model(state) = model(group_action @ state).
    """
    model = tm.CatchEquiModel_()
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
        inv_ball, inv_pad = apply_fn(model_params, inv_obs, inv_action)
        inv_pred = model.dist_to_obs(inv_ball, inv_pad).reshape(10, 5)
        transform_pred = catch_transform(pred).reshape(10, 5)

        comparison = pretty_compare(transform_pred, inv_pred)
        if not comparison:
            # uncomment lines 35, 37 to see when goes wrong
            # print()
            print(obs.reshape(10, 5))
        sol.append(comparison)

    print("Fraction of Predictions that are equivarant", sum(sol) / len(sol))
    assert all(sol), "Not all next state predictions are equivariant"
