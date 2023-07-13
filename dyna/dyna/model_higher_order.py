from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax.training.train_state import TrainState
from meta_rl.mutli_seed_script import Actions, Obs, Trajectory
from training import DynaHyperParams

EnvModelLosses = Any


class SASTuple(NamedTuple):
    state: Obs
    action: Actions
    next_state: Obs


def make_transition_model_update(hyper_params: DynaHyperParams, apply_fn):
    mini_batch_fn = make_mini_batch_fn(apply_fn)

    def tm_update_fn(
        rng: jt.PRNGKeyArray,
        train_state: TrainState,
        trajectories: Trajectory,
    ) -> Tuple[TrainState, EnvModelLosses]:
        no_mini_batch = hyper_params.M_NUM_MINIBATCHES
        data = trajectory_to_sas_tuple(trajectories)
        perm = jax.random.permutation(rng, data.state.shape[0])
        data = jax.tree_map(lambda x: x.at[perm].get(), data)
        batched_data = jax.tree_map(
            lambda x: x.reshape(no_mini_batch, -1, *x.shape[2:]), data
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

    return tm_update_fn


def make_mini_batch_fn(apply_fn):
    def loss_fn(model_params, sas_tuple):
        state, action, next_state = sas_tuple
        pred_next_state = apply_fn(model_params, state, action)
        return jnp.mean((pred_next_state - next_state) ** 2)

    def _mini_batch_fn(train_state, data):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(train_state.params, data)
        train_state = train_state.apply_gradients(grads)
        return train_state, loss

    return _mini_batch_fn


def trajectory_to_sas_tuple(trajectory: Trajectory) -> SASTuple:
    return SASTuple(
        state=trajectory.obs[:-1].reshape(-1, trajectory.obs.shape[2:]),
        action=trajectory.action.reshape(-1, trajectory.action.shape[2:]),
        next_state=trajectory.obs[1:].reshape(-1, trajectory.obs.shape[2:]),
    )
