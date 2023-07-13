from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from base_rl.models import ActorCritic
from base_rl.wrappers import LogWrapper
from flax.training.train_state import TrainState
from gymnax.environments.classic_control import CartPole
from model_based.nn_model import NNCartpole

from dyna.ac_higher_order import make_actor_critic_update
from dyna.model_higher_order import make_transition_model_update
from dyna.types import DynaHyperParams, DynaRunnerState, DynaState


def make_dyna_train_fn(dyna_hyp: DynaHyperParams):
    actor_critic = ActorCritic(1)

    tx_ac = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.ac_hyp.LR, eps=1e-5),
    )
    tx_model = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.model_hyp.LR, eps=1e-5),
    )

    def train(rng):
        env_model = LogWrapper(NNCartpole())
        env = LogWrapper(CartPole())
        rng, rng_ac, rng_reset, rng_model, rng_state = jax.random.split(rng, 5)
        ac_train_state = TrainState.create(
            apply_fn=jax.vmap(actor_critic.apply, in_axes=(None, 0)),
            params=actor_critic.init(rng_ac, jnp.ones((4))),
            tx=tx_ac,
        )
        model_train_state = TrainState.create(
            apply_fn=jax.vmap(env_model.transition_model.apply, in_axes=(None, 0, 0)),
            params=env_model.transition_model.init(  # type: ignore
                rng_model, jnp.ones((4)), jnp.ones((1))
            ),
            tx=tx_model,
        )

        model_free_update_fn = make_actor_critic_update(
            dyna_hyp,
            env,
            ac_train_state.apply_fn,
            False,
        )
        model_based_update_fn = make_actor_critic_update(
            dyna_hyp,
            env_model,
            ac_train_state.apply_fn,
            True,
        )

        env_model_update_fn = make_transition_model_update(
            dyna_hyp,
            model_train_state.apply_fn,
        )

        rng_reset = jax.random.split(rng_reset, dyna_hyp.NUM_ENVS)
        first_obs, cp_env_state = jax.vmap(env.reset, in_axes=(0, None))(
            rng_reset, CartPole().default_params
        )
        dyna_runner_state = DynaRunnerState(
            cartpole_env_state=cp_env_state,
            model_params=model_train_state.params,
            train_state=ac_train_state,
            last_obs=first_obs,
            rng=rng_state,
        )

        def _experience_step(dyna_state: DynaState, _) -> Tuple[DynaState, Tuple]:
            dyna_runner_state, model_train_state = dyna_state

            #######
            dyna_runner_state, (losses, trajectories) = model_free_update_fn(
                dyna_runner_state, None
            )
            rng = dyna_runner_state.rng
            rng, rng_model = jax.random.split(rng)

            #######
            model_train_state, model_losses = env_model_update_fn(
                rng_model, model_train_state, trajectories
            )

            rng, rng_next = jax.random.split(rng)
            new_runner_state = DynaRunnerState(
                model_params=model_train_state.params,
                train_state=dyna_runner_state.train_state,
                cartpole_env_state=dyna_runner_state.cartpole_env_state,
                last_obs=dyna_runner_state.last_obs,
                rng=rng_next,
            )

            # #######
            # dyna_runner_state, (p_loss, p_trajectories) = model_based_update_fn(
            #     new_runner_state,
            #     None,
            # )
            p_loss = 0.0
            p_trajectories = None

            new_runner_state = DynaRunnerState(
                model_params=model_train_state.params,
                train_state=dyna_runner_state.train_state,
                cartpole_env_state=dyna_runner_state.cartpole_env_state,
                last_obs=dyna_runner_state.last_obs,
                rng=rng_next,
            )

            new_dyna_state = DynaState(new_runner_state, model_train_state)
            infos = (
                losses,
                trajectories,
                p_loss,
                p_trajectories,
                model_losses,
            )
            return new_dyna_state, infos

        final_dyna_runner, infos = jax.lax.scan(
            _experience_step,
            DynaState(dyna_runner_state, model_train_state),
            None,
            length=dyna_hyp.NUM_UPDATES,
        )
        return final_dyna_runner, infos

    return train
