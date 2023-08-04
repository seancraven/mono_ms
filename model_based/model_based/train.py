from __future__ import annotations

import logging
import os
import pickle
import shutil
from abc import ABC
from math import prod
from typing import Any, NamedTuple, Tuple

import flax.linen as nn
import jax
import jaxtyping as jt
import optax
from base_rl.higher_order import Action, Observation
from base_rl.models import catch_transform
from flax.training.train_state import TrainState
from g_conv.c2 import C2Dense, C2DenseBinary
from jax import numpy as jnp
from optax import adam
from orbax import checkpoint

from model_based.sample_env import SARSDTuple

logger = logging.getLogger(__name__)


class LossData(NamedTuple):
    train_loss: jt.Array
    val_loss: jt.Array

    def flatten(self):
        return LossData(self.train_loss.reshape(-1), self.val_loss.reshape(-1))


class DebugData(NamedTuple):
    """Per dimension loss data."""

    x_pos_loss: jt.Array
    x_vel_loss: jt.Array
    theta_pos_loss: jt.Array
    theta_vel_loss: jt.Array

    def flatten(self) -> DebugData:
        return DebugData(
            self.x_pos_loss.reshape(-1),
            self.x_vel_loss.reshape(-1),
            self.theta_pos_loss.reshape(-1),
            self.theta_vel_loss.reshape(-1),
        )

    def as_array(self) -> jt.Array:
        return jnp.concatenate(
            [
                self.x_pos_loss,
                self.x_vel_loss,
                self.theta_pos_loss,
                self.theta_vel_loss,
            ],
            axis=-1,
        )


class TransitionModel(nn.Module, ABC):
    state_dim: int
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        ...


class Model(TransitionModel):
    @nn.compact
    def __call__(self, state: Observation, action: Observation) -> Observation:
        action = jnp.array((action,))
        state_embedding = nn.sigmoid(nn.Dense(self.hidden_dim)(state))
        action_embedding = nn.sigmoid(nn.Dense(self.hidden_dim)(action))
        concat = jnp.concatenate(
            [state_embedding.squeeze(), action_embedding.squeeze()], axis=0
        )
        hidden = nn.sigmoid(nn.Dense(self.hidden_dim)(concat))
        hidden = nn.sigmoid(nn.Dense(self.hidden_dim)(hidden))
        next_state = nn.Dense(self.state_dim)(hidden)

        return next_state


def proximal_state(state, next_states):
    assert next_states.shape[-1] == 2, f"{next_states.shape}"
    normal_state = next_states.at[..., 0].get()
    morror_state = next_states.at[..., 1].get()
    distance_next = ((normal_state.squeeze() - state.squeeze()) ** 2).sum()
    distance_mirror = ((morror_state.squeeze() - state.squeeze()) ** 2).sum()
    next_is_proximal = distance_next < distance_mirror
    return jax.lax.cond(next_is_proximal, lambda: normal_state, lambda: morror_state)


class EquiModel(TransitionModel):
    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        action = jnp.array((action,))
        state_embedding = nn.tanh(C2Dense(self.hidden_dim // 2)(state))
        action_embedding = nn.tanh(C2DenseBinary(self.hidden_dim // 2)(action))

        concat = jnp.concatenate(
            [state_embedding.reshape(-1), action_embedding.reshape(-1)], axis=0
        )

        hidden = nn.tanh(C2Dense(self.hidden_dim)(concat))
        hidden = nn.tanh(C2Dense(self.hidden_dim // 2)(hidden.reshape(-1)))
        hidden = C2Dense(self.state_dim)(hidden.reshape(-1))
        next_state = proximal_state(state, hidden)
        return next_state


class CatchModel(TransitionModel):
    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        action = jnp.array((action,))
        state_embedding = nn.relu(nn.Dense(self.hidden_dim // 2))(state)
        action_embedding = nn.relu(nn.Dense(self.hidden_dim // 2))(action)

        concat = jnp.concatenate([state_embedding, action_embedding], axis=0)
        hidden = nn.relu(nn.Dense(self.hidden_dim)(concat))
        hidden = nn.relu(nn.Dense(self.hidden_dim)(hidden))
        next_state = nn.Dense(self.state_dim)(hidden)
        return next_state


def catch_action_transform(action: Action) -> Action:
    action_inv = jax.lax.select(action == 1, jnp.array(1), jnp.abs(2 - action))
    return action_inv


class CatchEquiModel(TransitionModel):
    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        action = jnp.array((action,))
        state_embedding = nn.relu(
            C2Dense(self.hidden_dim // 2, transform=catch_transform)(state)
        )


class HyperParams(NamedTuple):
    batch_size: int = 256
    learning_rate: float = 1e-5
    train_frac: float = 0.8
    hidden_dim: int = 64
    epochs: int = 100
    model: Any = Model

    def get_train_size(self, data: SARSDTuple) -> int:
        """Returns the number of samples to use for training."""
        batch_count = self.get_batch_count(data)
        train_size = int(batch_count * self.batch_size)
        return train_size

    def get_batch_count(self, data: SARSDTuple) -> int:
        data_size = prod(data.reward.shape)  # type: ignore
        return int((data_size * self.train_frac) // self.batch_size)


def expand_scalar(x: jt.Array) -> jt.Array:
    if x.ndim == 1:
        return x.reshape((-1, 1))
    return x


def make_mse_loss_fn(apply_fn):
    def _loss_fn(
        params: jt.PyTree, sarsd_tuple: SARSDTuple
    ) -> Tuple[jt.Array, DebugData]:
        state, action, _, next_state, _ = sarsd_tuple
        next_state_pred = apply_fn(params, state, action)

        next_state_loss = jnp.mean((next_state - next_state_pred) ** 2, axis=0)

        debug_loss = DebugData(
            next_state_loss[0],
            next_state_loss[1],
            next_state_loss[2],
            next_state_loss[3],
        )

        # reward_loss = jnp.mean(bce_from_logit(reward_pred_logit, reward))
        # done_loss = jnp.mean(bce_from_logit(done_pred_logit, done))

        return next_state_loss.mean(), debug_loss  # + reward_loss + done_loss

    return _loss_fn


def make_bce_loss_fn(apply_fn):
    def _loss_fn(
        params: jt.PyTree, sarsd_tuple: SARSDTuple
    ) -> Tuple[jt.Array, DebugData]:
        state, action, _, next_state, _ = sarsd_tuple
        next_state_pred_logits = apply_fn(params, state, action)
        return (
            jnp.mean(
                optax.sigmoid_binary_cross_entropy(next_state_pred_logits, next_state)
            ),
            None,
        )

    return _loss_fn


def make_accuracy_loss_fn(apply_fn):
    def _loss_fn(
        params: jt.PyTree, sarsd_tuple: SARSDTuple
    ) -> Tuple[jt.Array, DebugData]:
        state, action, _, next_state, _ = sarsd_tuple
        next_state_pred_logits = apply_fn(params, state, action)
        next_state_pred = jnp.round(nn.softmax(next_state_pred_logits)).astype(
            jnp.int32
        )

        jax.debug.breakpoint()
        return jnp.mean(jnp.logical_and(next_state_pred, next_state)), None


def make_train(
    hyper_params: HyperParams,
    data: SARSDTuple,
    val_data: SARSDTuple,
    loss_function_ho=make_mse_loss_fn,
    val_loss_function_ho=make_mse_loss_fn,
):
    # non_term_index = (data.done == 0).squeeze()
    # data = jax.tree_map(lambda x: x.at[non_term_index, ...].get(), data)
    train_size = hyper_params.get_train_size(data)

    train_data, _ = data.partition(train_size)

    # non_term_index = (val_data.done == 0).squeeze()
    # val_data = jax.tree_map(lambda x: x.at[non_term_index, ...].get(), val_data)

    def train(rng, train_data, val_data):
        rng = jax.random.PRNGKey(42)
        train_size = hyper_params.get_train_size(data)
        batch_count = hyper_params.get_batch_count(data)
        state_dim = data.state.shape[-1]
        action_dim = 1
        network = hyper_params.model(state_dim, action_dim, hyper_params.hidden_dim)

        optimizer = adam(hyper_params.learning_rate)

        _, params_key = jax.random.split(rng)
        params = network.init(
            params_key,
            jnp.ones((state_dim,)),
            jnp.ones((action_dim,)),
        )
        apply_network = jax.vmap(network.apply, in_axes=(None, 0, 0))

        train_state = TrainState.create(
            apply_fn=apply_network,
            params=params,
            tx=optimizer,
        )
        _loss_fn = loss_function_ho(train_state.apply_fn)
        val_loss_fn = val_loss_function_ho(train_state.apply_fn)

        logger.debug(f"state.shape: {data.state.shape}")
        logger.debug(f"state_dim: {state_dim}")
        logger.debug(f"action_dim: {action_dim}")

        def _epoch(
            joint_state: Tuple[jt.PRNGKeyArray, TrainState, SARSDTuple, SARSDTuple], _
        ) -> Tuple[
            Tuple[jt.PRNGKeyArray, TrainState, SARSDTuple, SARSDTuple],
            Tuple[LossData, DebugData, DebugData],
        ]:
            def _mini_batch(
                train_state: TrainState, mini_batch: Any
            ) -> Tuple[TrainState, Tuple[jt.Array, DebugData]]:
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                (train_loss, debug_data), grad = grad_fn(train_state.params, mini_batch)
                train_state = train_state.apply_gradients(grads=grad)

                return train_state, (train_loss, debug_data)

            rng, train_state, train_data, val_data = joint_state
            _, rng = jax.random.split(rng)

            indecies = jax.random.permutation(rng, train_size)

            shuffle_train_data = jax.tree_map(
                lambda x: (x.at[indecies, ...].get()).reshape(
                    (batch_count, hyper_params.batch_size, -1)
                ),
                train_data,
            )

            (train_state, (train_loss, debug_data)) = jax.lax.scan(
                _mini_batch,
                train_state,
                shuffle_train_data,
            )
            val_loss, val_debug_data = val_loss_fn(train_state.params, val_data)

            return (rng, train_state, train_data, val_data), (
                LossData(train_loss, val_loss),
                debug_data,
                val_debug_data,
            )

        final_state, losses = jax.lax.scan(
            _epoch, (rng, train_state, train_data, val_data), None, hyper_params.epochs
        )

        return final_state, losses

    return lambda x: train(x, train_data, val_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hyper_params = HyperParams(model=EquiModel)
    data = pickle.load(open("replay_buffer.pickle", "rb"))
    len_data = prod(data.reward.shape)
    data = jax.tree_map(lambda x: x.reshape((len_data, -1)), data)
    train_data, val_data = data.partition(hyper_params.train_frac(data))
    train = make_train(hyper_params, train_data, val_data)
    final_state, losses = train(jax.random.PRNGKey(42))
    final_train_state = final_state[1]
    checkpointer = checkpoint.PyTreeCheckpointer()

    if os.path.exists("transition_model_tree/"):
        shutil.rmtree("transition_model_tree/")
    checkpointer.save("transition_model_tree/", final_train_state.params)

    train_loss, debug_loss, val_debug_loss = losses
    m_name = hyper_params.model.__name__
    jnp.save(
        f"transition_{m_name}_train_loss.npy",
        train_loss.train_loss.flatten(),
    )
    jnp.save(
        f"transition_{m_name}_val_loss.npy",
        train_loss.val_loss.flatten(),
    )
    pickle.dump(
        debug_loss,
        open(f"transition_{m_name}_debug_loss.pickle", "wb"),
    )
    pickle.dump(
        val_debug_loss,
        open(f"transition_{m_name}_val_debug_loss.pickle", "wb"),
    )
