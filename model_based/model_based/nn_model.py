import jaxtyping as jt
from gymnax.environments.classic_control import cartpole
from gymnax.environments.environment import EnvParams, EnvState
from optax._src.linear_algebra import lax
from orbax import checkpoint

from model_based.learn_trainsition_dynamics import Model


class NNCartpole(cartpole.CartPole):
    def __init__(self):
        super().__init__()
        self.TransitionModel = Model(*self.obs_shape, 1, 64)

        path = "/home/sean/ms_mono/model_based/transition_model_tree/"
        self.model_params = checkpoint.PyTreeCheckpointer().restore(path)

    @property
    def default_params(self):
        return cartpole.EnvParams()

    def step_env(
        self,
        _: jt.PRNGKeyArray,
        env_state: EnvState,
        action: jt.Array,
        params: EnvParams,
    ):
        prev_terminal = self.is_terminal(env_state, params)  # type: ignore
        reward = 1 - prev_terminal
        obs_ = self.get_obs(env_state)  # type: ignore
        next_env_obs = self.TransitionModel.apply(self.model_params, obs_, action)
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


def state_from_obs(obs: jt.Array, time_step: int) -> EnvState:
    return cartpole.EnvState(
        obs[0],  # type: ignore
        obs[1],  # type: ignore
        obs[2],  # type: ignore
        obs[3],  # type: ignore
        time_step,
    )
