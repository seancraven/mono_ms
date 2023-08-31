from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import gymnax
import jax.numpy as jnp
import jaxtyping as jt
from base_rl.higher_order import Actions, Obs, Params
from flax.training.train_state import TrainState
from model_based.train import Model

EnvModelLosses = Any


class SASTuple(NamedTuple):
    state: Obs
    action: Actions
    next_state: Obs
    done: Any

    def join(self, other: SASTuple) -> SASTuple:
        return SASTuple(
            state=jnp.concatenate([self.state, other.state], axis=0),
            action=jnp.concatenate([self.action, other.action], axis=0),
            next_state=jnp.concatenate([self.next_state, other.next_state], axis=0),
            done=jnp.concatenate([self.done, other.done], axis=0),
        )


class ActorCriticHyperParams(NamedTuple):
    """Hyper parameters for the actor critic model."""

    NUM_UPDATES: int = 1
    NUM_EPOCHS: int = 4
    MINIBATCH_SIZE: int = 128
    PRIV_NUM_TIMESTEPS: int = 128

    CLIP_EPS: float = 0.2
    VF_COEF: float = 0.5
    ENT_COEF: float = 0.01
    LR: float = 3.5e-4


class TransitionModelHyperParams(NamedTuple):
    USE_MODEL: bool = True
    MODEL_FN: Callable = Model
    MINIBATCH_SIZE: int = 64
    NUM_EPOCHS: int = 50
    LR: float = 1e-4
    PARAMS: Optional[jt.PyTree] = None


class DynaHyperParams(NamedTuple):
    ac_hyp: ActorCriticHyperParams = ActorCriticHyperParams()
    model_hyp: TransitionModelHyperParams = TransitionModelHyperParams()
    NUM_UPDATES: int = 1
    NUM_ENVS: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    MAX_GRAD_NORM: float = 0.5
    PLANNING_RATIO: float = 1.0

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

    @property
    def MODEL_HYP(self) -> DynaHyperParams:
        ac_hyp = ActorCriticHyperParams(
            NUM_UPDATES=int(self.ac_hyp.NUM_UPDATES * self.PLANNING_RATIO),
            NUM_EPOCHS=self.ac_hyp.NUM_EPOCHS,
            MINIBATCH_SIZE=self.ac_hyp.MINIBATCH_SIZE,
            PRIV_NUM_TIMESTEPS=self.ac_hyp.PRIV_NUM_TIMESTEPS,
            CLIP_EPS=self.ac_hyp.CLIP_EPS,
            VF_COEF=self.ac_hyp.VF_COEF,
            ENT_COEF=self.ac_hyp.ENT_COEF,
            LR=self.ac_hyp.LR,
        )
        return DynaHyperParams(
            ac_hyp=ac_hyp,
            model_hyp=self.model_hyp,
            NUM_UPDATES=self.NUM_UPDATES,
            NUM_ENVS=self.NUM_ENVS,
            GAMMA=self.GAMMA,
            GAE_LAMBDA=self.GAE_LAMBDA,
            MAX_GRAD_NORM=self.MAX_GRAD_NORM,
        )


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
