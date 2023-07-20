"""File to sample random trajectories from the environment."""
from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import NamedTuple, Tuple

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
from base_rl.models import ConvActorCritic
from gymnax import EnvState
from orbax import checkpoint


class SARSDTuple(NamedTuple):
    state: jt.Array
    action: jt.Array
    reward: jt.Array
    next_state: jt.Array
    done: jt.Array

    def partition(self, first_partition) -> Tuple[SARSDTuple, SARSDTuple]:
        one = SARSDTuple(
            self.state[:first_partition],
            self.action[:first_partition],
            self.reward[:first_partition],
            self.next_state[:first_partition],
            self.done[:first_partition],
        )
        two = SARSDTuple(
            self.state[first_partition:],
            self.action[first_partition:],
            self.reward[first_partition:],
            self.next_state[first_partition:],
            self.done[first_partition:],
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
    env_name: str, train_length: int
) -> Callable[[jt.PRNGKeyArray], SARSDTuple]:
    env, env_params = gymnax.make(env_name)
    policy_net = ConvActorCritic(
        2,
    )

    params = checkpoint.PyTreeCheckpointer().restore(
        "/home/sean/ms_mono/meta_rl/expert_actor_critic_tree/"
    )
    # params for all random seeds are stacked.
    params = jax.tree_map(lambda x: x[0], params)

    def experince(key) -> SARSDTuple:
        inital_obs, env_state = env.reset(key, env_params)

        def _step(
            joint_state: Tuple[jt.Array, EnvState, jt.PRNGKeyArray],
            _,
        ) -> Tuple[Tuple[jt.Array, EnvState, jt.Array], SARSDTuple]:
            obs, env_state, rng = joint_state
            _, action_rng, step_rng = jax.random.split(rng, 3)
            sample_action_dist, _ = policy_net.apply(params, obs.reshape((1, 4)))
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
    experience_fn = make_experience_fn("CartPole-v1", 500)
    expert_experience_fn = make_expert_experience_fn("CartPole-v1", 500)
    with jax.disable_jit():
        expert_experience_fn(jax.random.PRNGKey(42))
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
