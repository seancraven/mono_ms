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
        self.model_params = checkpoint.PyTreeCheckpointer().restore(
            "trainsition_model_tree"
        )

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
        next_env_state = self.TransitionModel.apply(
            self.model_params, env_state, action
        )
        t_stp = env_state.time + 1
        done = self.is_terminal(next_env_state, params)  # type: ignore
        state = cartpole.EnvState(
            next_env_state[0],  # type: ignore
            next_env_state[1],  # type: ignore
            next_env_state[2],  # type: ignore
            next_env_state[3],  # type: ignore
            t_stp,
        )
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},  # type: ignore
        )
