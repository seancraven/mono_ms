"""File to sample random trajectories from the environment."""
from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import NamedTuple, Tuple

import gymnax
import jax
import jaxtyping as jt
from gymnax import EnvState


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


ReplayBuffer = SARSDTuple


def make_experience_fn(
    env_name: str, train_length: int
) -> Callable[[jt.PRNGKeyArray], ReplayBuffer]:
    env, env_params = gymnax.make(env_name)

    def experince(key) -> ReplayBuffer:
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


def scan(f, carry, iterate):
    ys = []
    for i in iterate:
        carry, y = f(carry, i)
        y.append(y)
    return carry, jax.numpy.stack(ys)


if __name__ == "__main__":
    experience_fn = make_experience_fn("CartPole-v1", 500)
    with jax.disable_jit():
        experience_fn(jax.random.PRNGKey(42))
    vmap_experience_fn = jax.vmap(experience_fn, in_axes=0, out_axes=0)
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, 1000)
    replay_buffer = vmap_experience_fn(rngs)
    pickle.dump(replay_buffer, open("replay_buffer.pickle", "wb"))
    replay_buffer = pickle.load(open("replay_buffer.pickle", "rb"))
