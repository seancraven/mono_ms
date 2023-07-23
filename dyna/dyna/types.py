from __future__ import annotations

from copy import copy
from typing import Any, Callable, NamedTuple, Optional, Tuple

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
from base_rl.higher_order import Actions, Obs, Params, Transition
from flax.struct import dataclass, field
from flax.training.train_state import TrainState
from model_based.train import Model

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

    @property
    def no_transitions(self) -> int:
        return self.state.shape[0]

    @classmethod
    def from_transition(cls, transition: Transition) -> SASTuple:
        return cls(
            state=transition.obs.at[:-1].get(),
            action=transition.action.at[:-1].get(),
            next_state=transition.obs.at[1:].get(),
        )

    def flatten(self) -> SASTuple:
        return SASTuple(
            state=self.state.reshape((-1, 4)),
            action=self.action.reshape((-1, 1)),
            next_state=self.next_state.reshape((-1, 4)),
        )

    def get(self) -> SASTuple:
        return SASTuple(
            state=self.state.at[:].get(),
            action=self.action.at[:].get(),
            next_state=self.next_state.at[:].get(),
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
    USE_MODEL: bool = True
    MODEL_FN: Callable = Model
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
    MAX_GRAD_NORM: float = 0.5
    PLANNING_RATIO: float = 1.0

    @property
    def BUF_SIZE(self) -> int:
        return (
            self.AC_NUM_TRANSITIONS
            * self.NUM_UPDATES
            * self.NUM_ENVS
            * self.ac_hyp.NUM_UPDATES
        )

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
    replay_buffer: ReplayBuffer


class ReplayBuffer(NamedTuple):
    data: SASTuple
    insert_position: jt.Array = field(pytree_node=False)
    queue_size: int = field(pytree_node=False)

    @classmethod
    def create(cls, dummy_data: SASTuple, queue_size: int) -> ReplayBuffer:
        null_data = SASTuple(
            state=jnp.zeros((queue_size,) + dummy_data.state.shape),
            action=jnp.zeros((queue_size, 1), dtype=jnp.int32),
            next_state=jnp.zeros((queue_size,) + dummy_data.next_state.shape),
        )
        return ReplayBuffer(
            data=null_data,
            insert_position=jnp.zeros((), dtype=jnp.int32),
            queue_size=queue_size,
        )

    def insert(self, update_data: SASTuple) -> ReplayBuffer:
        update_data = update_data.flatten()

        insert_pos = self.insert_position
        buf_dim = self.queue_size
        update_dim = update_data.no_transitions

        total_no_transitions = insert_pos + update_dim
        roll = jnp.minimum(0, buf_dim - total_no_transitions)  # truthy if non 0
        data = jax.lax.cond(
            roll,
            lambda: jax.tree_map(lambda x: jnp.roll(x, roll, axis=0), self.data),
            lambda: self.data,
        )
        position = roll + insert_pos

        new_data = jax.tree_map(
            lambda x, y: jax.lax.dynamic_update_slice_in_dim(
                x,
                y,
                position,
                axis=0,
            ),
            data,
            update_data,
        )

        # Update the positions
        new_insert_pos = (position + update_dim) % (buf_dim + 1)

        return ReplayBuffer(
            data=new_data,
            insert_position=new_insert_pos,
            queue_size=buf_dim,
        )

    def sample(self, key) -> Tuple[ReplayBuffer, SASTuple]:
        """Samples Uniformally from the replay buffer."""
        idx = jax.random.randint(
            key,
            (self.data.no_transitions,),
            minval=0,
            maxval=self.insert_position,
        )
        sample = jax.tree_map(lambda x: x.at[idx].get(), self.data)
        new_buffer = ReplayBuffer(
            data=self.data.get(),
            insert_position=self.insert_position,
            queue_size=self.queue_size,
        )
        return new_buffer, sample
