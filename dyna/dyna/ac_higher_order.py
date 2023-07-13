from typing import Any, Callable, NamedTuple, Tuple, Union

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
from distrax import Categorical
from flax.training.train_state import TrainState
from meta_rl.mutli_seed_script import (BatchData, Obs, Params,
                                       PerTimestepScalar, Scalar, Trajectory,
                                       Transition)
from model_based.nn_model import NNCartpoleParams

from dyna.training import (ActorCriticHyperParams, DynaHyperParams,
                           DynaRunnerState)


class PartialLosses(NamedTuple):
    value_loss: jt.Array
    loss_actor: jt.Array
    entropy: jt.Array


class Losses(NamedTuple):
    total_loss: jt.Array
    partial_losses: PartialLosses


def make_actor_critic_update(
    dyna_hyp: DynaHyperParams,
    env_creation_function: Callable[[], gymnax.environments.CartPole],
    apply_fn: Callable[[Params, jt.Array], Tuple[Categorical, Obs]],
    model_based: bool = False,
) -> Callable[
    [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple[Losses, Trajectory]]
]:
    # Throught training constant functions.
    env = env_creation_function()
    _mini_batch_update_fn = make_ac_mini_batch_update_fn(apply_fn, dyna_hyp.ac_hyp)
    _gae_fn = make_gae_fn(dyna_hyp)

    def actor_critic_update(
        runner_state: DynaRunnerState,
        env_params: Union[gymnax.EnvParams, NNCartpoleParams],
    ) -> Tuple[DynaRunnerState, Tuple[Losses, Trajectory]]:
        """Upates the train state of the actor critic model, on a given model.

        Notes: Model Is frozen don't need to update.
        Notes: Must Return trajectory.
        """
        # Changes depending on learned transition dynmaics.
        _env_step_fn = make_env_step_fn(
            dyna_hyp, env, env_params, model_based=model_based
        )

        def _model_free_update(runner_state: DynaRunnerState, _):
            intermediate_runner_state, trajectories = jax.lax.scan(
                _env_step_fn,
                runner_state,  # type: ignore
                None,
                length=dyna_hyp.AC_NUM_TRANSITIONS,
            )
            (
                model_params,
                train_state,
                final_env_state,
                last_obs,
                rng,
            ) = intermediate_runner_state
            _, last_val = train_state.apply_fn(train_state.params, last_obs)
            advantages, targets = _gae_fn(trajectories, last_val)  # type: ignore

            def _ac_epoch(train_state_rng: Tuple[TrainState, jt.PRNGKeyArray], _):
                train_state, _rng = train_state_rng
                _, _rng = jax.random.split(_rng)
                batch_size = dyna_hyp.AC_BATCH_SIZE
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (trajectories, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                mini_batches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [dyna_hyp.AC_NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, losses = jax.lax.scan(
                    _mini_batch_update_fn,
                    train_state,
                    mini_batches,
                )

                return (train_state, _rng), losses

            final_train_state_rng, losses = jax.lax.scan(
                _ac_epoch,
                (train_state, rng),
                None,
                length=dyna_hyp.ac_hyp.NUM_EPOCHS,
            )

            final_train_state, rng = final_train_state_rng

            final_runner_state = DynaRunnerState(
                model_params,
                final_train_state,
                final_env_state,
                last_obs,
                rng,
            )
            return final_runner_state, (losses, trajectories)

        final_runner_state, metrics = jax.lax.scan(
            _model_free_update,
            runner_state,
            None,
            length=dyna_hyp.ac_hyp.NUM_UPDATES,
        )

        return final_runner_state, metrics  # type: ignore

    return actor_critic_update


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
    dyna_hyp: DynaHyperParams,
    env: gymnax.environments.CartPole,
    env_params,
    model_based: bool = False,
):
    def _env_step_model(
        runner_state: DynaRunnerState, _
    ) -> Tuple[DynaRunnerState, Transition]:
        model_params, train_state, env_state, last_obs, rng = runner_state

        rng, _rng = jax.random.split(rng)
        pi, value = train_state.apply_fn(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, dyna_hyp.NUM_ENVS)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None, None)
        )(
            rng_step,
            env_state,
            action,
            env_params,
            model_params,
        )
        transition = Transition(
            done,
            action,
            value,
            reward,
            log_prob,
            last_obs,
            info,
        )
        runner_state = DynaRunnerState(
            model_params,
            train_state,
            env_state,
            obsv,
            rng,
        )
        return runner_state, transition

    def _env_step_(
        runner_state: DynaRunnerState, _
    ) -> Tuple[DynaRunnerState, Transition]:
        model_params, train_state, env_state, last_obs, rng = runner_state

        rng, _rng = jax.random.split(rng)
        pi, value = train_state.apply_fn(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, dyna_hyp.NUM_ENVS)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(
            rng_step,
            env_state,
            action,
            env_params,
        )
        transition = Transition(
            done,
            action,
            value,
            reward,
            log_prob,
            last_obs,
            info,
        )
        runner_state = DynaRunnerState(
            model_params,
            train_state,
            env_state,
            obsv,
            rng,
        )
        return runner_state, transition

    if model_based:
        return _env_step_model
    return _env_step_
