from typing import Callable, Tuple

import distrax
import jax.numpy as jnp
from distrax import Categorical
from flax import linen as nn
from flax.linen.initializers import he_normal
from g_conv.c2 import C2Conv, C2Dense
from jaxtyping import Array


class BaseCritic(nn.Module):
    act: Callable
    h_dim: int = 64
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        critic = self.act(
            nn.Dense(
                self.h_dim,
                kernel_init=he_normal(),
                use_bias=self.bias,
            )(x)
        )
        critic = self.act(
            nn.Dense(
                self.h_dim,
                kernel_init=he_normal(),
                use_bias=self.bias,
            )(critic)
        )
        critic = nn.Dense(1, kernel_init=he_normal(), use_bias=True)(critic)
        return critic.squeeze()


Critic = lambda h_dim: BaseCritic(act=nn.relu, h_dim=h_dim)


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


class EquivariantActorCritic(nn.Module):
    a_dim: int
    h_dim: int = 64
    act: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        """
        Network equivarint to -1 transformation.
        """
        actor_mean = self.act(
            nn.Dense(self.h_dim, kernel_init=he_normal(), use_bias=False)(x)
        )
        actor_mean = self.act(
            nn.Dense(self.h_dim, kernel_init=he_normal(), use_bias=False)(actor_mean)
        )
        actor_mean = C2Dense(self.a_dim // 2)(actor_mean)
        pi = distrax.Categorical(logits=actor_mean.squeeze())

        critic = Critic(self.h_dim)(x)
        return pi, critic


def catch_transform(input):
    input_shape = input.shape
    input = input.reshape((-1, 10, 5))
    transformation = jnp.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    out = input @ transformation
    return out.reshape(input_shape)


def hidden_transform(input):
    input_shape = input.shape
    input = input.reshape((-1, 2))
    out = jnp.roll(input, axis=-1, shift=1)
    return out.reshape(input_shape)


class EquivariantCatchActorCritic(nn.Module):
    a_dim: int
    h_dim: int = 32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        actor_mean = self.act(
            C2Dense(self.h_dim, transform=catch_transform, use_bias=True)(x)
        )
        actor_mean = actor_mean.reshape(-1)

        actor_mean = C2Dense(self.h_dim, transform=hidden_transform, use_bias=True)(
            actor_mean
        )
        actor_mean = self.act(actor_mean.reshape(-1))

        actor_mean = C2Dense(1, transform=hidden_transform, use_bias=True)(actor_mean)

        actor_mean_0 = (
            C2Dense(1, transform=hidden_transform, use_bias=True)(actor_mean)
            .mean()
            .reshape(1)
        )
        logits = jnp.concatenate(
            [actor_mean[0].reshape((1,)), actor_mean_0, actor_mean[1].reshape((1,))]
        )

        pi = distrax.Categorical(logits=logits.squeeze())

        critic = Critic(self.h_dim)(x)
        return pi, critic


class ActorCritic(nn.Module):
    a_dim: int
    h_dim: int = 64
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        actor_mean = self.act(nn.Dense(self.h_dim, kernel_init=he_normal())(x))
        actor_mean = self.act(nn.Dense(self.h_dim, kernel_init=he_normal())(actor_mean))
        actor_mean = nn.Dense(self.a_dim, kernel_init=he_normal())(actor_mean)
        pi = distrax.Categorical(logits=actor_mean.squeeze())

        critic = Critic(self.h_dim)(x)
        return pi, critic


class ConvEquivariantActorCritic(nn.Module):
    """Equiariant Convolutional Actor Critic Network."""

    a_dim: int
    h_dim: int = 64
    act: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        actor_mean = self.act(C2Conv(features=self.h_dim, kernel_size=((1,)))(x))
        actor_mean = self.act(
            C2Conv(features=self.h_dim // 2, kernel_size=((1,)))(actor_mean)
        )
        actor_mean = C2Conv(features=self.a_dim, kernel_size=((1,)))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean.squeeze())

        critic = Critic(self.h_dim)(x)
        return pi, critic


class ConvActorCritic(nn.Module):
    a_dim: int
    h_dim: int = 64
    act: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Conv(
            features=self.h_dim,
            kernel_size=((1,)),
            kernel_init=he_normal(),
        )(x)
        actor_mean = self.act(actor_mean)
        actor_mean = nn.Conv(
            features=self.h_dim,
            kernel_size=((1,)),
            kernel_init=he_normal(),
        )(actor_mean)
        actor_mean = self.act(actor_mean)
        actor_mean = nn.Conv(
            features=self.a_dim,
            kernel_size=((1,)),
            kernel_init=he_normal(),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean.squeeze())

        critic = Critic(self.h_dim)(x)

        return pi, critic
