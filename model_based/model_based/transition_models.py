from __future__ import annotations

import logging
from abc import ABC
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jaxtyping as jt
from base_rl.higher_order import Action, Observation
from base_rl.models import catch_transform, hidden_transform
from g_conv.c2 import C2Dense, C2DenseBinary
from jax import numpy as jnp

logger = logging.getLogger(__name__)


class TransitionModel(nn.Module, ABC):
    state_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        ...


class Model(TransitionModel):
    @nn.compact
    def __call__(self, state: Observation, action: Observation) -> Observation:
        action = jnp.array((action,))
        state_embedding = nn.relu(nn.Dense(self.hidden_dim)(state))
        action_embedding = nn.relu(nn.Dense(self.hidden_dim)(action))
        concat = jnp.concatenate(
            [state_embedding.squeeze(), action_embedding.squeeze()], axis=0
        )
        hidden = nn.relu(nn.Dense(self.hidden_dim)(concat))
        hidden = nn.relu(nn.Dense(self.hidden_dim)(hidden))
        next_state = nn.Dense(self.state_dim)(hidden)

        return next_state


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


class BaseCatchModel(ABC, nn.Module):
    state_dim: int = 50
    hidden_dim: int = 64

    def __call__(
        self, state, action
    ) -> Tuple[distrax.Categorical, distrax.Categorical]:
        raise NotImplementedError

    @classmethod
    def dist_to_obs(
        cls, ball_dist: distrax.Categorical, paddel_dist: distrax.Categorical
    ) -> Observation:
        ball = jnp.zeros_like(ball_dist.probs).at[ball_dist.mode()].set(1)
        paddle = jnp.zeros_like(paddel_dist.probs).at[paddel_dist.mode()].set(1)
        total = jnp.concatenate([ball.reshape(-1), paddle.reshape(-1)], axis=0)
        return total


def catch_action_transform(action: Action) -> Action:
    action_inv = jax.lax.cond(
        (action == 1).all(),  # action is only one dim have to call all to turn to bool
        lambda: jnp.array(1.0, dtype=jnp.float32).reshape(1),
        lambda: jnp.abs(2 - action).reshape(1).astype(jnp.float32),
    )
    return action_inv


class CatchModel(BaseCatchModel):
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


class SimpleCatchInv(BaseCatchModel):
    """Invariant transition Model.

    The model produces two distributions that are related to each other by the group
    inversion action for cartpole.
    """

    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        state_embedding_layer = C2Dense(
            self.hidden_dim // 2,
            transform=catch_transform,
        )
        action_embedding_layer = C2Dense(
            self.hidden_dim // 2,
            transform=catch_action_transform,
        )
        hidden_layer = nn.Dense(self.hidden_dim)
        hidden_layer_2 = nn.Dense(self.hidden_dim)
        out_layer = nn.Dense(self.state_dim)
        # hidden_layer = C2Dense(self.hidden_dim, transform=hidden_transform)
        # hidden_layer_2 = C2Dense(self.hidden_dim, transform=hidden_transform)
        # out_layer = C2Dense(self.state_dim)

        ## State/ Action embedding are equivariant to group action.
        ## the application of group action permutes the -1 axis.
        state_embedding = nn.relu(state_embedding_layer(state))
        action_embedding = nn.relu(action_embedding_layer(action))

        ## Take the mean along the embeddings to make it invariant embedding.
        invariant_embedding = jnp.concatenate(
            [state_embedding.mean(axis=-1), action_embedding.mean(axis=-1)], axis=0
        )

        hidden = nn.relu(hidden_layer(invariant_embedding))
        hidden = nn.relu(hidden_layer_2(hidden))

        logits = out_layer(hidden)
        # Transform the invariant output state.
        logits_inv = catch_transform(logits)
        stacked_logits = jnp.stack(
            [logits.reshape(-1), logits_inv.reshape(-1)], axis=-1
        )

        return stacked_logits


class SimpleCatchEqui(BaseCatchModel):
    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        state_embedding_layer = C2Dense(
            self.hidden_dim // 2,
            transform=catch_transform,
        )
        action_embedding_layer = C2Dense(
            self.hidden_dim // 2,
            transform=catch_action_transform,
        )
        hidden_layer = C2Dense(self.hidden_dim, transform=hidden_transform)
        hidden_layer_2 = C2Dense(self.hidden_dim, transform=hidden_transform)
        out_layer = C2Dense(self.state_dim, transform=hidden_transform)

        state_embedding = nn.relu(state_embedding_layer(state))
        action_embedding = nn.relu(action_embedding_layer(action))

        concat = jnp.concatenate([state_embedding, action_embedding], axis=0).reshape(
            -1
        )
        hidden = nn.relu(hidden_layer(concat))
        hidden = nn.relu(hidden_layer_2(hidden.reshape(-1)))
        out = out_layer(hidden.reshape(-1))
        stacked_logits = convert_group_action(out)
        return stacked_logits


def convert_group_action(stacked_logits):
    idn_logits = stacked_logits.at[..., 0].get()
    inv_logits = stacked_logits.at[..., 1].get()
    inv_logits = catch_transform(inv_logits)
    return jnp.stack([idn_logits, inv_logits], axis=-1)


class CatchEquiModel(BaseCatchModel):
    """Currently Nearly Equivariant model apart from some edge cases."""

    @nn.compact
    def __call__(self, state: Observation, action: Action):
        action = action.reshape(1)

        stacked_logits = SimpleCatchInv(self.state_dim, self.hidden_dim)(state, action)

        logits = catch_pool(stacked_logits, state)

        ball_dist = distrax.Categorical(logits=logits.at[:45].get())
        pad_dist = distrax.Categorical(logits=logits.at[45:].get())
        return ball_dist, pad_dist


class CatchEquiModel_(BaseCatchModel):
    @nn.compact
    def __call__(self, state, action):
        action = action.reshape(1)

        stacked_logits = SimpleCatchEqui(self.state_dim, self.hidden_dim)(state, action)

        logits = catch_pool(stacked_logits, state)

        ball_dist = distrax.Categorical(logits=logits.at[:45].get())
        pad_dist = distrax.Categorical(logits=logits.at[45:].get())
        return ball_dist, pad_dist


def catch_pool(stacked_logits, state) -> jt.Array:
    """Pooling layer, to make prediction equivariant.

    Assumes that it is given two distributions for the next state of catch related by
    the group action of catch.
    stacked_logits[0] = catch_transform(stacked_logits[1])
    where stacked_logits[0] describes one of the two possible predicted next states.
    Finds which logits describe a state that is closer to the initial state (state).
    """

    def _ball_l1(pred_logits: jt.Array, state: jt.Array):
        """Find L1 distance between predicted state ball and previous state."""
        pred_ball_y, pred_ball_x = divmod(
            distrax.Categorical(logits=pred_logits).mode(), 5
        )
        ball_loc_y, ball_loc_x = divmod(distrax.Categorical(logits=state).mode(), 5)

        return jnp.sqrt(
            (pred_ball_x - ball_loc_x) ** 2 + (pred_ball_y - ball_loc_y) ** 2
        )

    def _paddel_l1(pred_logits: jt.Array, state: jt.Array):
        """Find L1 distance between predicted state paddle and previous state."""
        assert pred_logits.shape == state.shape
        pad_loc = distrax.Categorical(logits=state).mode()
        pred_pad = distrax.Categorical(logits=pred_logits).mode()
        return jnp.abs(pred_pad - pad_loc)

    def joint_l1(logits, initial_state):
        """Sum of l1 distance of paddle and ball.

        The paddle l1 distance is scaled by a factor of 0.1 to make all
        possible distances unique.
        """
        init_ball, init_pad = initial_state.at[:45].get(), initial_state.at[45:].get()
        ball_logits, pad_logits = logits.at[:45].get(), logits.at[45:].get()
        ball_l1 = _ball_l1(ball_logits, init_ball)
        pad_l1 = _paddel_l1(pad_logits, init_pad)
        return ball_l1 + pad_l1 * 0.1

    pred_logits = stacked_logits.at[..., 0].get()
    pred_logits_inv = stacked_logits.at[..., 1].get()

    l1 = joint_l1(pred_logits, state)
    l1_inv = joint_l1(pred_logits_inv, state)

    def distance_compare():
        return jax.lax.cond(
            l1 < l1_inv,
            lambda: pred_logits,
            lambda: pred_logits_inv,
        )

    return distance_compare()

    # return jax.lax.cond(
    #     l1 == l1_inv,
    #     lambda: jnp.mean(stacked_logits, axis=-1),
    #     lambda: distance_compare(),
    # )
