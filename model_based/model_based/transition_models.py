from __future__ import annotations

import logging
from abc import ABC
from math import dist
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jaxtyping as jt
import numpy as np
from base_rl.higher_order import Action, Observation
from base_rl.models import catch_transform, hidden_transform
from g_conv.c2 import C2Dense, C2DenseBinary
from jax import numpy as jnp

logger = logging.getLogger(__name__)


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


def proximal_state_pool(state: jt.Array, next_states: jt.Array) -> Observation:
    """Pools returning the closes state to the initial_state.

    Only implemented for the C2 group
    """
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
        next_state = proximal_state_pool(state, hidden)
        return next_state


class CatchModel(nn.Module):
    state_dim: int = 50
    action_dim: int = 3
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self, state: Observation, action: Action
    ) -> Tuple[distrax.Categorical, distrax.Categorical]:
        action = jnp.array((action,))
        state_embedding = nn.relu(nn.Dense(self.hidden_dim // 2)(state))
        action_embedding = nn.relu(nn.Dense(self.hidden_dim // 2)(action))

        concat = jnp.concatenate(
            [state_embedding.squeeze(), action_embedding.squeeze()], axis=0
        )
        hidden = nn.relu(nn.Dense(self.hidden_dim)(concat))
        hidden = nn.relu(nn.Dense(self.hidden_dim)(hidden))
        next_state_ball_logit = nn.Dense(45)(hidden)
        next_state_paddel_logit = nn.Dense(5)(hidden)
        ball_dist = distrax.Categorical(logits=next_state_ball_logit)
        paddel_dist = distrax.Categorical(logits=next_state_paddel_logit)
        return ball_dist, paddel_dist

    @classmethod
    def dist_to_obs(
        cls, ball_dist: distrax.Categorical, paddel_dist: distrax.Categorical
    ) -> Observation:
        ball = jnp.zeros_like(ball_dist.probs).at[ball_dist.mode()].set(1)
        paddle = jnp.zeros_like(paddel_dist.probs).at[paddel_dist.mode()].set(1)
        total = jnp.concatenate([ball.reshape(-1), paddle.reshape(-1)], axis=0)
        return total


def catch_action_transform(action: Action) -> Action:
    action_inv = jax.lax.select(
        action == 1, jnp.array(1).reshape(1), jnp.abs(2 - action).reshape(1)
    )
    return action_inv


def catch_proximal_state_pool(
    stacked_states: jt.Array, initial_state: jt.Array
) -> jt.Array:
    """Pools returning the closes state to the initial_state.

    Only implemented for the C2 group
    """
    next_state = stacked_states.at[..., 0].get()
    mirror_state = stacked_states.at[..., 1].get()
    index_pad = np.arange(0, 5)
    index_ball = np.arange(0, 45)

    init_ball = jnp.sum(initial_state.at[:45].get() * index_ball) % 5
    init_pad = jnp.sum(initial_state.at[45:].get() * index_pad)

    next_state = next_state.reshape(50)
    ball_state, pad_state = next_state.at[:45].get(), next_state.at[45:].get()
    mirror_ball, mirror_pad = mirror_state.at[:45].get(), mirror_state.at[45:].get()

    ball_loc = distrax.Categorical(logits=ball_state).mode() % 5
    pad_loc = distrax.Categorical(logits=pad_state).mode()

    mirror_ball_loc = distrax.Categorical(logits=mirror_ball).mode() % 5
    mirror_pad_loc = distrax.Categorical(logits=mirror_pad).mode()

    next_dist = jnp.abs(ball_loc - init_ball) + jnp.abs(pad_loc - init_pad)
    mirror_dist = jnp.abs(mirror_ball_loc - init_ball) + jnp.abs(
        mirror_pad_loc - init_pad
    )

    # return next_state, mirror_state
    return jax.lax.cond(
        next_dist < mirror_dist,
        lambda: next_state.reshape(-1),
        lambda: mirror_state.reshape(-1),
    )


class _CatchEquiLogits(CatchModel):
    @nn.compact
    def __call__(self, state: Observation, action: Action) -> jt.Array:
        action = jnp.array((action,)).reshape(1)
        state_embedding_layer = C2Dense(self.hidden_dim // 2, transform=catch_transform)
        action_embedding_layer = C2Dense(
            self.hidden_dim // 2, transform=catch_action_transform
        )

        state_embedding = nn.relu(state_embedding_layer(state))
        action_embedding = nn.relu(action_embedding_layer(action))

        concat = jnp.concatenate(
            [state_embedding.reshape(-1), action_embedding.reshape(-1)], axis=0
        )
        hidden_layer = C2Dense(self.hidden_dim, transform=hidden_transform)
        hidden_layer_2 = C2Dense(self.state_dim, transform=hidden_transform)

        hidden = nn.relu(hidden_layer(concat.reshape(-1)))
        hidden = nn.relu(hidden_layer_2(hidden.reshape(-1)))
        return hidden


class CatchEquiModel(CatchModel):
    """Equivariant catch transition model."""

    @nn.compact
    def __call__(
        self, state: Observation, action: Action
    ) -> Tuple[distrax.Categorical, distrax.Categorical]:
        next_states_stacked = _CatchEquiLogits(
            self.state_dim, self.hidden_dim, self.action_dim
        )(state, action)
        next_state = catch_proximal_state_pool(next_states_stacked, state)
        ball_dist = distrax.Categorical(logits=next_state.at[..., :45].get())
        pad_dist = distrax.Categorical(logits=next_state.at[..., 45:].get())

        return ball_dist, pad_dist
