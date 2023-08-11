from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxtyping as jt
from gymnax.environments.bsuite import catch
from gymnax.environments.classic_control import cartpole
from gymnax.environments.environment import EnvParams, EnvState
from optax._src.linear_algebra import lax

from model_based.transition_models import BaseCatchModel, TransitionModel


class NNCartpole(cartpole.CartPole):
    """A neural network model of the cartpole environment.

    The model default params returns the default params of the cartpole environment.
    """

    def __init__(self, model=TransitionModel):
        super().__init__()
        self.transition_model = model(*self.obs_shape, 64)

    @property
    def default_params(self) -> EnvParams:
        return cartpole.EnvParams()  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jt.PRNGKeyArray,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
        model_params: Optional[jt.PyTree] = None,
    ) -> Tuple[jt.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params, model_params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(
        self,
        _: jt.PRNGKeyArray,
        env_state: cartpole.EnvState,
        action: jt.Array,
        params: cartpole.EnvParams,
        model_params: jt.PyTree,
    ):
        assert model_params is not None, "Model params must be provided"
        prev_terminal = self.is_terminal(env_state, params)
        reward = 1 - prev_terminal
        obs_ = self.get_obs(env_state)  # type: ignore
        next_env_obs = self.transition_model.apply(model_params, obs_, action)
        time_step = env_state.time + 1
        next_env_state = self.state_from_obs(next_env_obs, time_step)  # type: ignore

        done = self.is_terminal(next_env_state, params)  # type: ignore
        return (
            lax.stop_gradient(next_env_obs),
            lax.stop_gradient(next_env_state),
            reward,
            done,
            {"discount": self.discount(next_env_state, params)},  # type: ignore
        )

    def observation_space(self, params):
        return super().observation_space(params)

    def state_from_obs(self, obs: jt.Array, time_step: int) -> EnvState:
        return cartpole.EnvState(
            obs[0],  # type: ignore
            obs[1],  # type: ignore
            obs[2],  # type: ignore
            obs[3],  # type: ignore
            time_step,
        )


class NNCartpoleParams(NamedTuple):
    model: jt.PyTree
    cartpole: cartpole.EnvParams = cartpole.EnvParams()


class NNCatch(catch.Catch):
    def __init__(self, model=BaseCatchModel):
        super().__init__()
        obs_dim = self.rows * self.columns
        self.transition_model = model(obs_dim, 64)

    @property
    def default_params(self) -> EnvParams:
        return catch.EnvParams()  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jt.PRNGKeyArray,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
        model_params: Optional[jt.PyTree] = None,
    ) -> Tuple[jt.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params, model_params  # type: ignore
        )
        obs_re, state_re = self.reset_env(key_reset, params)  # type: ignore
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(
        self,
        key: jt.PRNGKeyArray,
        env_state: catch.EnvState,
        action: jt.Array,
        params: catch.EnvParams,
        model_params: jt.PyTree,
    ):
        assert model_params is not None, "Model params must be provided"
        prev_done = env_state.prev_done
        time_step = env_state.time + 1

        reset_ball_x, reset_ball_y, reset_pad_x, reset_pad_y = catch.sample_init_state(
            key, self.rows, self.columns
        )
        reset_state = catch.EnvState(
            reset_ball_x, reset_ball_y, reset_pad_x, reset_pad_y, False, time_step
        )

        obs_ = self.get_obs(env_state)
        ball_dist, paddle_dist = self.transition_model.apply(
            model_params, obs_.reshape(-1), action
        )
        pred_next_env_state = self.state_from_dist(ball_dist, paddle_dist, time_step)  # type: ignore

        next_env_state = jax.lax.cond(
            prev_done, lambda: reset_state, lambda: pred_next_env_state
        )

        caught = next_env_state.ball_x == next_env_state.paddle_x
        reward = next_env_state.prev_done * jax.lax.select(caught, 1.0, -1.0)

        done = self.is_terminal(next_env_state, params)  # type: ignore

        return (
            lax.stop_gradient(self.get_obs(next_env_state)),
            lax.stop_gradient(next_env_state),
            reward,
            done,
            {"discount": self.discount(next_env_state, params)},  # type: ignore
        )

    def state_from_dist(self, ball_dist, paddle_dist, time_step: int):
        PADDLE_Y = self.columns - 1
        FINAL_ROW_OF_BALL = PADDLE_Y - 1
        ball_x, ball_y = jnp.divmod(ball_dist.mode(), self.rows)
        paddle_x = paddle_dist.mode()
        paddle_y = PADDLE_Y
        done = ball_y == FINAL_ROW_OF_BALL
        return catch.EnvState(
            ball_x,
            ball_y,
            paddle_x,
            paddle_y,
            done,
            time_step,
        )
