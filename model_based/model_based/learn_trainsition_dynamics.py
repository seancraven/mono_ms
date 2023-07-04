import logging
import pickle
from typing import NamedTuple, Tuple

import flax.linen as nn
import jax
import jaxtyping as jt
from flax.training.train_state import TrainState
from jax import numpy as jnp
from optax import adam

from model_based.sample_env import ReplayBuffer, SARSDTuple

logger = logging.getLogger(__name__)
TRAIN_FRAC = 0.8


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
        action_embedding = nn.relu(nn.Dense(self.hidden_dim)(action))
        concat = jnp.concatenate([state_embedding, action_embedding], axis=0)
        hidden = nn.relu(nn.Dense(self.hidden_dim)(concat))
        next_state = nn.Dense(self.state_dim)(hidden)
        reward = nn.softmax(nn.Dense(1)(hidden))
        done = nn.softmax(nn.Dense(1)(hidden))

        return (next_state, reward, done)


def main():
    data: ReplayBuffer = pickle.load(open("replay_buffer.pickle", "rb"))
    partitioned_data = jax.tree_map(
        lambda x: (
            x[: int(x.shape[0] * TRAIN_FRAC)],
            x[int(x.shape[0] * TRAIN_FRAC) :],
        ),
        data,
    )
    state_dim = data.state.shape[-1]
    action_dim = 1
    hidden_dim = 64
    logger.debug(f"state_dim: {state_dim}")
    logger.debug(f"action_dim: {action_dim}")
    network = Model(state_dim, action_dim, hidden_dim)
    optimizer = adam(1e-3)
    rng = jax.random.PRNGKey(42)
    _, params_key = jax.random.split(rng)
    params = network.init(params_key, jnp.ones((state_dim,)), jnp.ones((action_dim,)))
    apply_network = jax.vmap(network.apply, in_axes=(None, 0, 0), out_axes=0)
    TrainState.create(
        apply_fn=apply_network,
        params=params,
        tx=optimizer,
    )

    def _epoch(
        joint_state: Tuple[jt.PRNGKeyArray, TrainState], _
    ) -> Tuple[TrainState, LossData]:
        def _mini_batch(
            train_state: TrainState,
            batch: Tuple[SARSDTuple, SARSDTuple],
        ) -> Tuple[TrainState, LossData]:
            train_batch, val_batch = batch

            def _loss_fn(sarsd_tuple: SARSDTuple) -> jt.Array:
                state, action, reward, next_state, done, _ = sarsd_tuple
                (next_state_pred, reward_pred, done_pred) = network(state, action)
                next_state_loss = jnp.mean((next_state - next_state_pred) ** 2)
                reward_loss = jnp.mean((reward - reward_pred) ** 2)
                done_loss = jnp.mean(
                    (done * jnp.log(done_pred)) + ((1 - done) * jnp.log(1 - done_pred))
                )
                return next_state_loss + reward_loss + done_loss

            grad_fn = jax.value_and_grad(_loss_fn)

            loss, grad = grad_fn(train_batch)
            val_loss = _loss_fn(val_batch)
            train_state = train_state.apply_gradients(grads=grad)

            return train_state, LossData(train_loss=loss, val_loss=val_loss)

        rng, train_state = joint_state
        rng, batch_key = jax.random.split(rng)

        jax.random.permutation(batch_key, data.state.shape[0])

        return state, losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
