from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from base_rl.higher_order import Trajectory
from flax.training.train_state import TrainState
from model_based.train import SARSDTuple

from dyna.types import DynaHyperParams, EnvModelLosses, SASTuple


def make_transition_model_update(hyper_params: DynaHyperParams, apply_fn):
    mini_batch_fn = make_mini_batch_fn(apply_fn)
    # exp_fn = make_experience_fn("CartPole-v1", 10_000)
    # exp_fn = jax.jit(exp_fn)
    # exp = exp_fn(jax.random.PRNGKey(0))
    # base_sas = sarsd_to_sas_tuple(exp)

    def pass_update_fn(
        rng: jt.PRNGKeyArray,
        train_state: TrainState,
        trajectories: Trajectory,
    ) -> Tuple[TrainState, EnvModelLosses]:
        return train_state, None

    def tm_update_fn(
        rng: jt.PRNGKeyArray,
        train_state: TrainState,
        trajectories: Trajectory,
    ) -> Tuple[TrainState, EnvModelLosses]:
        no_mini_batch = hyper_params.M_NUM_MINIBATCHES
        data = trajectory_to_sas_tuple(trajectories)
        # data = data.join(base_sas)
        perm = jax.random.permutation(rng, data.state.shape[0])
        data = jax.tree_map(lambda x: x.at[perm].get(), data)
        batched_data = jax.tree_map(
            lambda x: x.reshape(no_mini_batch, -1, x.shape[-1]), data
        )

        def _epoch(
            train_state: TrainState,
            _,
        ) -> Tuple[TrainState, EnvModelLosses]:
            state, losses = jax.lax.scan(mini_batch_fn, train_state, batched_data)
            return state, losses

        train_state, losses = jax.lax.scan(
            _epoch, train_state, None, length=hyper_params.model_hyp.NUM_EPOCHS
        )
        return train_state, losses

    return tm_update_fn if hyper_params.model_hyp.NUM_EPOCHS > 0 else pass_update_fn


def make_mini_batch_fn(apply_fn):
    def loss_fn(model_params, sas_tuple):
        state, action, next_state, done = sas_tuple
        pred_next_state = apply_fn(model_params, state, action)
        return jnp.mean((1-done)*(pred_next_state - next_state) ** 2)

    def _mini_batch_fn(train_state, data):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(train_state.params, data)

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    return _mini_batch_fn


def trajectory_to_sas_tuple(trajectory: Trajectory) -> SASTuple:
    len_ = np.prod(trajectory.done.shape)
    state = trajectory.obs.reshape(len_, -1)
    action = trajectory.action.reshape(len_, 1)
    done = trajectory.done.reshape(len_, -1)
    next_state = jnp.roll(state, shift=1, axis=0)
    return SASTuple(state, action, next_state)


def sarsd_to_sas_tuple(sarsd: SARSDTuple) -> SASTuple:
    len_ = np.prod(sarsd.done.shape)
    return SASTuple(
        state=sarsd.state.reshape(len_, -1).at[:-1, ...].get(),
        action=sarsd.action.reshape(len_, -1).at[:-1, ...].get(),
        next_state=sarsd.next_state.reshape(len_, -1).at[1:, ...].get(),
        done=sarsd.done.reshape(len_, -1).at[:-1, ...].get()
    )
