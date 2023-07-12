from typing import Callable, NamedTuple, Tuple

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
from distrax import Categorical
from flax.training.train_state import TrainState
from meta_rl.mutli_seed_script import (BatchData, Obs, Params,
                                       PerTimestepScalar, RunnerState, Scalar,
                                       Trajectory, Transition)

from dyna.training import ActorCriticHyperParams, DynaHyperParams


class PartialLosses(NamedTuple):
    value_loss: jt.Array
    loss_actor: jt.Array
    entropy: jt.Array


class Losses(NamedTuple):
    total_loss: jt.Array
    partial_losses: PartialLosses


def make_gae_fn(
    dyna_hyp: DynaHyperParams,
) -> Callable[[Trajectory, Scalar], Tuple[PerTimestepScalar, PerTimestepScalar]]:
    def _calculate_gae(
        traj_batch: Trajectory,
        last_val: Scalar,
    ) -> Tuple[PerTimestepScalar, PerTimestepScalar]:
        """Recursively calculates truncated GAE."""

        def _get_advantages(
            gae_and_next_value: Tuple[jt.Array, jt.Array],
            transition: Transition,
        ) -> Tuple[Tuple[jt.Array, jt.Array], jt.Array]:
            """One step of GAE."""
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + dyna_hyp.GAMMA * next_value * (1 - done) - value
            gae = delta + dyna_hyp.GAMMA * dyna_hyp.GAE_LAMBDA * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,  # type: ignore
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    return _calculate_gae


def make_ac_mini_batch_update_fn(
    apply_fn: Callable[[Params, Obs], Tuple[Categorical, PerTimestepScalar]],
    config: ActorCriticHyperParams,
) -> Callable[[TrainState, BatchData], Tuple[TrainState, Losses]]:
    def _update_minbatch(
        train_state: TrainState, batch_info: BatchData
    ) -> Tuple[TrainState, Losses]:
        """Loss evaluated and Adam applied to a single minibatch."""
        traj_batch, advantages, targets = batch_info

        def _loss_fn(
            params: Params,
            traj_batch: Trajectory,
            gae: PerTimestepScalar,
            targets: PerTimestepScalar,
        ) -> Tuple[jt.Array, Tuple[jt.Array, jt.Array, jt.Array]]:
            pi, value = apply_fn(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)  # type: ignore

            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                -config.CLIP_EPS, config.CLIP_EPS
            )
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config.CLIP_EPS,
                    1.0 + config.CLIP_EPS,
                )
                * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy
            )

            return Losses(total_loss, PartialLosses(value_loss, loss_actor, entropy))

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss

    return _update_minbatch


def make_env_step_fn(
    dyna_hyp: DynaHyperParams, env: gymnax.environments.CartPole, env_params
):
    def _env_step(runner_state: RunnerState, _) -> Tuple[RunnerState, Transition]:
        train_state, env_state, last_obs, rng = runner_state

        rng, _rng = jax.random.split(rng)
        pi, value = train_state.apply_fn(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, dyna_hyp.NUM_ENVS)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, action, env_params)
        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info  # type: ignore
        )
        runner_state = (train_state, env_state, obsv, rng)  # type: ignore
        return runner_state, transition

    return _env_step
