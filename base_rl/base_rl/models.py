from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from distrax import Categorical
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from g_conv.c2 import C2Conv, C2Dense, C2DenseLift
from jaxtyping import Array


class ACSequential(nn.Module):
    """Sequential Actor Critic Network.

    Call function returns a tuple of (action_logits, state_value).

    Args:
        actor_layers: A tuple of actor layers.
        critic_layers: A tuple of critic layers.
    __call__ Args:
        state: The state of the environment.
    __call__ Returns:
        A tuple of (action_logits, state_value).
    """

    actor_layers: Tuple[nn.Module, ...]
    critic_layers: Tuple[nn.Module, ...]

    @nn.compact
    def __call__(self, state: Array) -> Tuple[Categorical, Array]:
        action_logits = state
        state_value = state
        for actor_layer, critic_layer in zip(self.actor_layers, self.critic_layers):
            state_value = critic_layer(state_value)
            action_logits = actor_layer(action_logits)
        return Categorical(logits=action_logits), state_value.squeeze()


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class EquivariantActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.relu(C2DenseLift(features=64)(x))
        actor_mean = nn.relu(C2Dense(features=64)(actor_mean))
        actor_mean = nn.relu(C2Dense(features=self.action_dim)(actor_mean))
        actor_mean = jax.lax.cond(
            (actor_mean.at[:, 0].get() > 0.0).all(),
            actor_mean.at[:, 0].get,
            actor_mean.at[:, 1].get,
        )
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.relu(C2DenseLift(features=64)(x))
        critic = nn.relu(C2Dense(features=64)(critic))
        critic = C2Dense(features=1)(critic)
        critic = jax.lax.cond(
            (critic.at[:, 0].get() > 0.0).all(),
            critic.at[:, 0].get,
            critic.at[:, 1].get,
        )

        return pi, critic.squeeze()


class EquivariantConvActorCritic(nn.Module):
    action_dim: int
    activaton: str = "tanh"

    ## TODO: Indexing of the layer kernel shapes needs to get fixed
    @nn.compact
    def __call__(self, x):
        activation = nn.relu
        actor_mean = C2Conv(features=64, kernel_size=((1,)))(x)
        actor_mean = activation(actor_mean)
        actor_mean = C2Conv(features=64, kernel_size=((1,)))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = C2Conv(features=self.action_dim, kernel_size=((1,)))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return pi, jnp.squeeze(critic, axis=-1)


class ConvActorCritic(nn.Module):
    action_dim: int
    activaton: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu
        actor_mean = nn.Conv(features=64, kernel_size=((1,)))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Conv(features=128, kernel_size=((1,)))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Conv(features=self.action_dim, kernel_size=((1,)))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return pi, jnp.squeeze(critic, axis=-1)
