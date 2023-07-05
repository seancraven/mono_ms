import logging
import pickle
from math import prod
from typing import Any, NamedTuple, Tuple

import flax.linen as nn
import jax
import jaxtyping as jt
import matplotlib.pyplot as plt
from flax.training.train_state import TrainState
from jax import numpy as jnp
from optax import adam

from model_based.sample_env import ReplayBuffer, SARSDTuple

logger = logging.getLogger(__name__)


class LossData(NamedTuple):
    train_loss: jt.Array
    val_loss: jt.Array


class Model(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state, action):
        state_embedding = nn.relu(nn.Dense(self.hidden_dim)(state))
        assert action.ndim > 0, action
        action_embedding = nn.relu(nn.Dense(self.hidden_dim)(action))
        concat = jnp.concatenate([state_embedding, action_embedding], axis=0)
        hidden = nn.relu(nn.Dense(self.hidden_dim)(concat))
        next_state = nn.Dense(self.state_dim)(hidden)
        reward = nn.softmax(nn.Dense(1)(hidden))
        done = nn.softmax(nn.Dense(1)(hidden))

        return (next_state, reward, done)


class HyperParams(NamedTuple):
    batch_size: int = 256
    learning_rate: float = 1e-3
    train_frac: float = 0.8
    hidden_dim: int = 64
    epochs: int = 100

    def get_train_size(self, data: ReplayBuffer) -> int:
        """Returns the number of samples to use for training."""
        batch_count = self.get_batch_count(data)
        train_size = int(batch_count * self.batch_size)
        return train_size

    def get_batch_count(self, data: ReplayBuffer) -> int:
        data_size = prod(data.reward.shape)  # type: ignore
        return int((data_size * self.train_frac) // self.batch_size)


def expand_scalar(x: jt.Array) -> jt.Array:
    if x.ndim == 1:
        return x.reshape((-1, 1))
    return x


def make_train(hyper_params: HyperParams):
    data: ReplayBuffer = pickle.load(open("replay_buffer.pickle", "rb"))
    data = SARSDTuple(*jax.tree_map(lambda x: x.astype(jnp.float32), data))

    train_size = hyper_params.get_train_size(data)
    flattened_data = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), data)
    train_data, val_data = flattened_data.partition(train_size)
    val_data = jax.tree_map(lambda x: expand_scalar(x), val_data)

    def train(rng, train_data, val_data):
        rng = jax.random.PRNGKey(42)
        train_size = hyper_params.get_train_size(data)
        batch_count = hyper_params.get_batch_count(data)
        state_dim = data.state.shape[-1]
        action_dim = 1
        network = Model(state_dim, action_dim, hyper_params.hidden_dim)

        optimizer = adam(hyper_params.learning_rate)

        _, params_key = jax.random.split(rng)
        params = network.init(
            params_key, jnp.ones((state_dim,)), jnp.ones((action_dim,))
        )
        apply_network = jax.vmap(network.apply, in_axes=(None, 0, 0), out_axes=0)

        train_state = TrainState.create(
            apply_fn=apply_network,
            params=params,
            tx=optimizer,
        )

        logger.debug(f"state.shape: {data.state.shape}")
        logger.debug(f"state_dim: {state_dim}")
        logger.debug(f"action_dim: {action_dim}")

        def _epoch(
            joint_state: Tuple[jt.PRNGKeyArray, TrainState, SARSDTuple, SARSDTuple], _
        ) -> Tuple[
            Tuple[jt.PRNGKeyArray, TrainState, SARSDTuple, SARSDTuple], LossData
        ]:
            def _loss_fn(params: jt.PyTree, sarsd_tuple: SARSDTuple) -> jt.Array:
                state, action, reward, next_state, done = sarsd_tuple
                (next_state_pred, reward_pred, done_pred) = train_state.apply_fn(
                    params, state, action
                )
                next_state_loss = jnp.mean((next_state - next_state_pred) ** 2)
                reward_loss = jnp.mean((reward - reward_pred) ** 2)
                done_loss = jnp.mean(
                    (done * jnp.log(done_pred)) + ((1 - done) * jnp.log(1 - done_pred))
                )
                jax.debug.breakpoint()
                return next_state_loss + reward_loss + done_loss

            def _mini_batch(
                train_state: TrainState, mini_batch: Any
            ) -> Tuple[TrainState, jt.Array]:
                grad_fn = jax.value_and_grad(_loss_fn)

                train_loss, grad = grad_fn(train_state.params, mini_batch)
                train_state = train_state.apply_gradients(grads=grad)

                return train_state, train_loss

            rng, train_state, train_data, val_data = joint_state
            _, rng = jax.random.split(rng)

            indecies = jax.random.permutation(rng, train_size)

            shuffle_train_data = jax.tree_map(
                lambda x: (x.at[indecies].get()).reshape(
                    (batch_count, hyper_params.batch_size, -1)
                ),
                train_data,
            )

            (train_state, train_loss) = jax.lax.scan(
                _mini_batch,
                train_state,
                shuffle_train_data,
            )
            val_loss = _loss_fn(train_state.params, val_data)

            return (rng, train_state, train_data, val_data), LossData(
                train_loss, val_loss
            )

        final_state, losses = jax.lax.scan(
            _epoch, (rng, train_state, train_data, val_data), None, hyper_params.epochs
        )
        return final_state, losses

    return lambda x: train(x, train_data, val_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hyper_params = HyperParams()
    with jax.disable_jit(False):
        train = make_train(hyper_params)
        final_state, losses = train(jax.random.PRNGKey(42))

    print(losses.train_loss.shape)
    plt.plot(losses.train_loss, label="train")
    plt.show()
