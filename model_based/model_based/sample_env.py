"""File to sample random trajectories from the environment."""
from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import NamedTuple, Tuple

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from base_rl.higher_order import ActorCritic
from gymnax import EnvState


class SARSDTuple(NamedTuple):
    state: jt.Array
    action: jt.Array
    reward: jt.Array
    next_state: jt.Array
    done: jt.Array

    def partition(self, first_partition) -> Tuple[SARSDTuple, SARSDTuple]:
        one = SARSDTuple(
            self.state.at[:first_partition, ...].get(),
            self.action.at[:first_partition, ...].get(),
            self.reward.at[:first_partition, ...].get(),
            self.next_state.at[:first_partition, ...].get(),
            self.done.at[:first_partition, ...].get(),
        )
        two = SARSDTuple(
            self.state.at[first_partition:, ...].get(),
            self.action.at[first_partition:, ...].get(),
            self.reward.at[first_partition:, ...].get(),
            self.next_state.at[first_partition:, ...].get(),
            self.done.at[first_partition:, ...].get(),
        )
        return one, two

    def join(self, other: SARSDTuple) -> SARSDTuple:
        return SARSDTuple(
            jnp.concatenate([self.state.squeeze(), other.state.squeeze()]),
            jnp.concatenate([self.action.squeeze(), other.action.squeeze()]),
            jnp.concatenate([self.reward.squeeze(), other.reward.squeeze()]),
            jnp.concatenate([self.next_state.squeeze(), other.next_state.squeeze()]),
            jnp.concatenate([self.done.squeeze(), other.done.squeeze()]),
        )

    def filter_by_action(self, act) -> SARSDTuple:
        idx = (self.action == act).squeeze()
        return SARSDTuple(
            self.state.at[idx, ...].get(),
            self.action.at[idx, ...].get(),
            self.reward.at[idx, ...].get(),
            self.next_state.at[idx, ...].get(),
            self.done.at[idx, ...].get(),
        )


def make_experience_fn(
    env_name: str, train_length: int
) -> Callable[[jt.PRNGKeyArray], SARSDTuple]:
    env, env_params = gymnax.make(env_name)

    def experince(key) -> SARSDTuple:
        inital_obs, env_state = env.reset(key, env_params)

        def _step(
            joint_state: Tuple[jt.Array, EnvState, jt.PRNGKeyArray],
            _,
        ) -> Tuple[Tuple[jt.Array, EnvState, jt.Array], SARSDTuple]:
            obs, env_state, rng = joint_state
            _, action_rng, step_rng = jax.random.split(rng, 3)
            sample_action = env.action_space(env_params).sample(action_rng)  # type: ignore
            next_obs, env_state, reward, done, _ = env.step(
                step_rng, env_state, sample_action
            )
            state_tuple = SARSDTuple(obs, sample_action, reward, next_obs, done)  # type: ignore
            return (next_obs, env_state, rng), state_tuple  # type: ignore

        _, replayBuffer = jax.lax.scan(
            _step, (inital_obs, env_state, key), None, length=train_length
        )
        return replayBuffer

    return experince


def make_expert_experience_fn(
    env_name: str, train_length: int, expert_params: jt.PyTree
) -> Callable[[jt.PRNGKeyArray], SARSDTuple]:
    env, env_params = gymnax.make(env_name)
    policy_net = ActorCritic(env.action_space(env_params).n)

    def experince(key) -> SARSDTuple:
        inital_obs, env_state = env.reset(key, env_params)

        def _step(
            joint_state: Tuple[jt.Array, EnvState, jt.PRNGKeyArray],
            _,
        ) -> Tuple[Tuple[jt.Array, EnvState, jt.Array], SARSDTuple]:
            obs, env_state, rng = joint_state
            _, action_rng, step_rng = jax.random.split(rng, 3)
            sample_action_dist, _ = policy_net.apply(
                expert_params,
                obs.reshape((1, np.prod(env.observation_space(env_params).shape))),
            )
            sample_action = sample_action_dist.sample(seed=action_rng)
            next_obs, env_state, reward, done, _ = env.step(
                step_rng, env_state, sample_action.squeeze()
            )
            state_tuple = SARSDTuple(obs, sample_action, reward, next_obs, done)  # type: ignore
            return (next_obs, env_state, rng), state_tuple  # type: ignore

        _, replayBuffer = jax.lax.scan(
            _step, (inital_obs, env_state, key), None, length=train_length
        )
        return replayBuffer

    return experince


if __name__ == "__main__":
    from base_rl.higher_order import CONFIG, make_train

    print("Training Expert")
    train_fn = jax.jit(make_train(CONFIG, ActorCritic))
    rng = jax.random.PRNGKey(42)

    result = train_fn(rng)
    train_state = result["runner_state"][0]
    expert_params = train_state.params

    experience_fn = make_experience_fn("CartPole-v1", 500)
    expert_experience_fn = make_expert_experience_fn("CartPole-v1", 500, expert_params)
    vmap_experience_fn = jax.vmap(experience_fn, in_axes=0, out_axes=0)
    vmap_expert_experience_fn = jax.vmap(expert_experience_fn, in_axes=0, out_axes=0)
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, 1000)
    simple_rng, expert_rng = rngs[:500], rngs[500:]
    replay_buffer = vmap_experience_fn(rngs)
    expert_replay_buffer = vmap_expert_experience_fn(rngs)
    replay_buffer = replay_buffer.join(expert_replay_buffer)
    pickle.dump(replay_buffer, open("replay_buffer.pickle", "wb"))
    replay_buffer = pickle.load(open("replay_buffer.pickle", "rb"))
