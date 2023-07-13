from __future__ import annotations

from typing import NamedTuple

import jaxtyping as jt
from gymnax.environments.classic_control import cartpole
from gymnax.environments.environment import EnvParams, EnvState
from optax._src.linear_algebra import lax

from model_based.train import Model


class NNCartpole(cartpole.CartPole):
    """
    A neural network model of the cartpole environment.

    The model default params returns the default params of the cartpole environment.
    """

    def __init__(self):
        super().__init__()
        self.transition_model = Model(*self.obs_shape, 1, 64)

    @property
    def default_params(self) -> EnvParams:
        return cartpole.EnvParams()  # type: ignore

    def step_env(
        self,
        _: jt.PRNGKeyArray,
        env_state: cartpole.EnvState,
        action: jt.Array,
        params: cartpole.EnvParams,
        model_params: jt.PyTree,
    ):
        prev_terminal = self.is_terminal(env_state, params)
        reward = 1 - prev_terminal
        obs_ = self.get_obs(env_state)  # type: ignore
        next_env_obs = self.transition_model.apply(model_params, obs_, action)
        time_step = env_state.time + 1
        next_env_state = state_from_obs(next_env_obs, time_step)  # type: ignore

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


class NNCartpoleParams(NamedTuple):
    model: jt.PyTree
    cartpole: cartpole.EnvParams = cartpole.EnvParams()


def state_from_obs(obs: jt.Array, time_step: int) -> EnvState:
    return cartpole.EnvState(
        obs[0],  # type: ignore
        obs[1],  # type: ignore
        obs[2],  # type: ignore
        obs[3],  # type: ignore
        time_step,
    )
