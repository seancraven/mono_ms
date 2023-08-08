from __future__ import annotations

import logging
from abc import ABC
from typing import Any, NamedTuple, Optional, Tuple

import distrax
import flax.linen as nn
import jax
from base_rl.higher_order import Action, Observation
from base_rl.models import catch_transform
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
        raise NotImplementedError
