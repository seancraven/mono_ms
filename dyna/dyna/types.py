from __future__ import annotations

from typing import Any, NamedTuple, Optional

import gymnax
import jax.numpy as jnp
import jaxtyping as jt
from base_rl.higher_order import Actions, Obs, Params
from flax.training.train_state import TrainState

EnvModelLosses = Any


class SASTuple(NamedTuple):
    state: Obs
    action: Actions
    next_state: Obs

    def join(self, other: SASTuple) -> SASTuple:
        return SASTuple(
            state=jnp.concatenate([self.state, other.state], axis=0),
            action=jnp.concatenate([self.action, other.action], axis=0),
            next_state=jnp.concatenate([self.next_state, other.next_state], axis=0),
        )


class ActorCriticHyperParams(NamedTuple):
    """Hyper parameters for the actor critic model."""

    NUM_UPDATES: int = 5
    NUM_EPOCHS: int = 6
    MINIBATCH_SIZE: int = 64
    PRIV_NUM_TIMESTEPS: int = 128

    CLIP_EPS: float = 0.2
    VF_COEF: float = 0.5
    ENT_COEF: float = 0.01
    LR: float = 2.5e-4


class TransitionModelHyperParams(NamedTuple):
    MINIBATCH_SIZE: int = 64
    NUM_EPOCHS: int = 10
    LR: float = 1e-3


class DynaHyperParams(NamedTuple):
    ac_hyp: ActorCriticHyperParams = ActorCriticHyperParams()
    model_hyp: TransitionModelHyperParams = TransitionModelHyperParams()
    NUM_UPDATES: int = 10
    NUM_ENVS: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95

    USE_MODEL: bool = True
    MAX_GRAD_NORM: float = 0.5

    @property
    def AC_NUM_TRANSITIONS(self) -> int:
        assert (
            self.ac_hyp.PRIV_NUM_TIMESTEPS * self.NUM_ENVS
        ) % self.ac_hyp.MINIBATCH_SIZE == 0
        return self.ac_hyp.PRIV_NUM_TIMESTEPS * self.NUM_ENVS

    @property
    def AC_BATCH_SIZE(self) -> int:
        return self.AC_NUM_TRANSITIONS

    @property
    def AC_NUM_TIMESTEPS(self) -> int:
        return self.AC_NUM_TRANSITIONS // self.NUM_ENVS

    @property
    def M_NUM_MINIBATCHES(self) -> int:
        return self.AC_NUM_TRANSITIONS // self.model_hyp.MINIBATCH_SIZE

    @property
    def AC_NUM_MINIBATCHES(self) -> int:
        return (self.NUM_ENVS * self.AC_NUM_TIMESTEPS) // self.ac_hyp.MINIBATCH_SIZE


class DynaRunnerState(NamedTuple):
    model_params: Optional[Params]
    train_state: TrainState
    cartpole_env_state: gymnax.EnvState
    last_obs: Obs
    rng: jt.PRNGKeyArray

    def get_train_state(self) -> TrainState:
        return self.train_state

    def get_env_state(self) -> gymnax.EnvState:
        return self.cartpole_env_state


class DynaState(NamedTuple):
    dyna_runner_state: DynaRunnerState
    model_train_state: TrainState
    model_env_state: gymnax.EnvState
