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
    action_inv = jax.lax.cond(
        (action == 1).all(),  # action is only one dim have to call all to turn to bool
        lambda: jnp.array(1.0, dtype=jnp.float32).reshape(1),
        lambda: jnp.abs(2 - action).reshape(1).astype(jnp.float32),
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


class _CatchEquiLogits(nn.Module):
    state_dim: int = 50
    hidden_dim: int = 64

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


class SimpleCatchInv(nn.Module):
    state_dim: int = 50
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, state: Observation, action: Action) -> Observation:
        action = jnp.array((action,)).reshape(1)
        state_embedding_layer = C2Dense(self.hidden_dim // 2, transform=catch_transform)
        action_embedding_layer = C2Dense(
            self.hidden_dim // 2, transform=catch_action_transform
        )

        state_embedding = nn.relu(state_embedding_layer(state))
        action_embedding = nn.relu(action_embedding_layer(action))

        invariant_embedding = jnp.concatenate(
            [state_embedding.mean(axis=-1), action_embedding.mean(axis=-1)], axis=0
        )
        hidden_layer = nn.Dense(self.hidden_dim)
        out_layer = nn.Dense(self.state_dim)
        hidden = nn.relu(hidden_layer(invariant_embedding))
        logits = out_layer(hidden)
        logits_inv = catch_transform(logits)
        stacked_logits = jnp.stack(
            [logits.reshape(-1), logits_inv.reshape(-1)], axis=-1
        )

        return stacked_logits


class SimpleCatchEqui(nn.Module):
    state_dim: int = 50
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, state: Observation, action: Action):
        action = action.reshape(1)

        stacked_logits = SimpleCatchInv(self.state_dim, self.hidden_dim)(state, action)

        logits = catch_pool(stacked_logits, state)

        ball_dist = distrax.Categorical(logits=logits.at[:45].get())
        pad_dist = distrax.Categorical(logits=logits.at[45:].get())
        return ball_dist, pad_dist

    @classmethod
    def dist_to_obs(
        cls, ball_dist: distrax.Categorical, paddel_dist: distrax.Categorical
    ) -> Observation:
        ball = jnp.zeros_like(ball_dist.probs).at[ball_dist.mode()].set(1)
        paddle = jnp.zeros_like(paddel_dist.probs).at[paddel_dist.mode()].set(1)
        total = jnp.concatenate([ball.reshape(-1), paddle.reshape(-1)], axis=0)
        return total


def catch_pool(stacked_logits, state) -> jt.Array:
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
        """Sum of l1 distance of paddle and ball."""
        init_ball, init_pad = initial_state.at[:45].get(), initial_state.at[45:].get()
        ball_logits, pad_logits = logits.at[:45].get(), logits.at[45:].get()
        ball_l1 = _ball_l1(ball_logits, init_ball)
        pad_l1 = _paddel_l1(pad_logits, init_pad)
        return ball_l1 + pad_l1

    pred_logits = stacked_logits.at[..., 0].get()
    pred_logits_inv = stacked_logits.at[..., 1].get()

    l1 = joint_l1(pred_logits, state)
    l1_inv = joint_l1(pred_logits_inv, state)
    return jax.lax.cond(
        l1 < l1_inv,
        lambda: pred_logits,
        lambda: pred_logits_inv,
    )


def _catch_pool(
    identity_tup: Tuple[jt.Array, ...], inv_tup: Tuple[jt.Array, ...]
) -> jt.Array:
    """Pools over next states to find closest to next_state.

    The G-CNN model on a forward pass produces two distributions, which are equivariant
    to the symmetry transformation of catch. To select between these in an equivariant
    manner, on a well trained network it would be sufficient to select the distribution
    which is closest to the initial_state, however, in early stages of training this is not the
    case. To find the correct branch 1) perform prediction with the network on the orgional state
    2) perform prediction with the network on the inversion of the sate.

    Args:
    ----
        identity_logits:
        dist A
        dist B

        inv_logits:
        dist B
        dist A

        identity_tup: Tuple of stacked logits shape (50, 2), and initial_state.
            The logits describe two distributions stacked, one of the predicted next
            state, and a null prediction. When a transformation is applied to the input,
            the null prediction and the predicted next state swap position.
        inv_tup: Tuple of stacked logits, and the initial_state with a inversion
        transformation applied. The logits in this case describe two distributions which
        are the inversion along the final axis to the identity logits, due to the equivariant
        nature of the network.

    Returns:
    -------



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
        """Sum of l1 distance of paddle and ball."""
        init_ball, init_pad = initial_state.at[:45].get(), initial_state.at[45:].get()
        ball_logits, pad_logits = logits.at[:45].get(), logits.at[45:].get()
        ball_l1 = _ball_l1(ball_logits, init_ball)
        pad_l1 = _paddel_l1(pad_logits, init_pad)
        return ball_l1 + pad_l1

    def pair_l1(stacked_logits, intial_state):
        next_state = stacked_logits.at[..., 0].get()
        mirror_state = stacked_logits.at[..., 1].get()
        return joint_l1(next_state, intial_state), joint_l1(mirror_state, intial_state)

    def min_dist(equi_pair, equi_pair_logits):
        """Is the state an inversion or orgional."""
        equi_0_l1 = joint_l1(equi_pair_logits[0], idn_state)
        equi_1_l1 = joint_l1(equi_pair_logits[1], idn_state)

        return jax.lax.cond(
            equi_0_l1 < equi_1_l1,
            lambda: equi_pair_logits[0],
            lambda: equi_pair_logits[1],
        )

    idn_logits, idn_state = identity_tup
    inv_logits, inv_state = inv_tup

    idn_pair_l1 = pair_l1(idn_logits, idn_state)
    inv_pair_l1 = pair_l1(inv_logits, inv_state)

    # How far pair a is from stat and mirror
    equi_pair_a_l1 = (idn_pair_l1[0], inv_pair_l1[1])
    equi_pair_a_logits = (idn_logits.at[..., 0].get(), inv_logits.at[..., 1].get())

    # How far pair b is from state and mirror
    equi_pair_b_l1 = (idn_pair_l1[1], inv_pair_l1[0])
    equi_pair_b_logits = (idn_logits.at[..., 1].get(), inv_logits.at[..., 0].get())

    logits = jax.lax.cond(
        sum(equi_pair_a_l1) < sum(equi_pair_b_l1),
        lambda: min_dist(equi_pair_a_l1, equi_pair_a_logits),
        lambda: min_dist(equi_pair_b_l1, equi_pair_b_logits),
    )

    return logits


class CatchPooler(nn.Module):
    state_dim: int = 50
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, state: Observation, action: Action) -> jt.Array:
        state_inv = catch_transform(state)
        action_inv = catch_action_transform(action)
        model = _CatchEquiLogits(self.state_dim, self.hidden_dim)
        logits = model(state, action)
        logits_inv = model(state_inv, action_inv)
        logits = catch_pool((logits, state), (logits_inv, state_inv))
        return logits


class CatchEquiModel(nn.Module):
    """Equivariant catch transition model."""

    state_dim: int = 50

    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self, state: Observation, action: Action
    ) -> Tuple[distrax.Categorical, distrax.Categorical]:
        logits = CatchPooler(self.state_dim, self.hidden_dim)(state, action)
        ball_dist = distrax.Categorical(logits=logits.at[..., :45].get())
        pad_dist = distrax.Categorical(logits=logits.at[..., 45:].get())

        return ball_dist, pad_dist

    @classmethod
    def dist_to_obs(
        cls, ball_dist: distrax.Categorical, paddel_dist: distrax.Categorical
    ) -> Observation:
        ball = jnp.zeros_like(ball_dist.probs).at[ball_dist.mode()].set(1)
        paddle = jnp.zeros_like(paddel_dist.probs).at[paddel_dist.mode()].set(1)
        total = jnp.concatenate([ball.reshape(-1), paddle.reshape(-1)], axis=0)
        return total
