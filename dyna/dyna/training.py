from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from base_rl.higher_order import Trajectory, Transition
from base_rl.models import ActorCritic
from base_rl.wrappers import LogWrapper
from flax.training.train_state import TrainState
from gymnax.environments.classic_control import CartPole
from jaxtyping import PRNGKeyArray
from model_based.nn_model import NNCartpole

from dyna.ac_higher_order import Losses, make_actor_critic_update
from dyna.model_higher_order import make_transition_model_update
from dyna.types import DynaHyperParams, DynaRunnerState, DynaState


def make_dyna_train_fn(dyna_hyp: DynaHyperParams):
    """Higher order function to make dyna training function configured
    with dyna hyperparams"""
    actor_critic = ActorCritic(2)

    tx_ac = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.ac_hyp.LR, eps=1e-5),
    )
    tx_model = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.model_hyp.LR, eps=1e-5),
    )

    def train(rng):
        env_model = LogWrapper(NNCartpole(model=dyna_hyp.model_hyp.MODEL_FN))
        env = LogWrapper(CartPole())

        assert env.action_space().n == env_model.action_space().n
        assert (
            env.observation_space(env.default_params).shape
            == env_model.observation_space(env_model.default_params).shape
        )
        state_shape = env.observation_space(env.default_params).shape
        action_shape = env.action_space().shape

        rng, rng_ac, rng_reset, rng_model, rng_state = jax.random.split(rng, 5)
        ac_train_state = TrainState.create(
            apply_fn=jax.vmap(actor_critic.apply, in_axes=(None, 0)),
            params=actor_critic.init(rng_ac, jnp.ones(state_shape)),
            tx=tx_ac,
        )

        model_train_state = TrainState.create(
            apply_fn=jax.vmap(env_model.transition_model.apply, in_axes=(None, 0, 0)),
            params=env_model.transition_model.init(
                rng_model, jnp.ones(state_shape), jnp.ones(action_shape)
            ),
            tx=tx_model,
        )
        print(
            "Model Params:",
            sum(x.size for x in jax.tree_util.tree_leaves(model_train_state.params)),
        )

        model_free_update_fn = make_actor_critic_update(
            dyna_hyp,
            env,
            ac_train_state.apply_fn,
            False,
        )
        model_based_update_fn = make_actor_critic_update(
            dyna_hyp.MODEL_HYP,
            env_model,
            ac_train_state.apply_fn,
            True,
        )

        env_model_update_fn = make_transition_model_update(
            dyna_hyp,
            model_train_state.apply_fn,
        )
        ex_step = make_experience_step(
            dyna_hyp.model_hyp.USE_MODEL,
            model_free_update_fn,
            model_based_update_fn,
            env_model_update_fn,
        )
        # gain_ex_fun = make_gain_exp(
        #     dyna_hyp,
        #     model_free_update_fn,
        #     model_based_update_fn,
        #     env_model_update_fn,
        # )
        # mf_ex_step = make_experience_step(
        #     False,
        #     model_free_update_fn,
        #     model_based_update_fn,
        #     env_model_update_fn,
        # )

        rng_reset = jax.random.split(rng_reset, dyna_hyp.NUM_ENVS)
        first_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            rng_reset, env.default_params
        )
        _, model_env_state = jax.vmap(env_model.reset, in_axes=(0, None))(
            rng_reset, env_model.default_params
        )
        dyna_runner_state = DynaRunnerState(
            cartpole_env_state=env_state,
            model_params=model_train_state.params,
            train_state=ac_train_state,
            last_obs=first_obs,
            rng=rng_state,
        )
        dyna_state = DynaState(
            dyna_runner_state,
            model_train_state,
            model_env_state,
        )
        # dyna_state, infos = mf_ex_step(
        #     dyna_state,
        #     None,
        # )
        # dyna_state, infos = gain_ex_fun(
        #     dyna_state,
        #     infos[1],
        # )
        # return dyna_state, infos

        final_dyna_runner, infos = jax.lax.scan(
            ex_step,
            dyna_state,
            None,
            length=dyna_hyp.NUM_UPDATES,
        )
        return final_dyna_runner, infos

    return train


def make_experience_step(
    use_model: bool,
    model_free_update_fn: Callable[
        [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple[Losses, Trajectory]]
    ],
    model_based_update_fn: Callable[
        [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple[Losses, Trajectory]]
    ],
    env_model_update_fn: Callable[
        [PRNGKeyArray, TrainState, Transition], Tuple[TrainState, Tuple]
    ],
) -> Callable[[DynaState, Any], Tuple[DynaState, Tuple]]:
    """
    Function factory for agent and/or model training steps.
    Returns a scannable function.
    """

    def _model_free_step(dyna_state: DynaState, _) -> Tuple[DynaState, Tuple]:
        dyna_runner_state, model_train_state, model_env_state, rp_buf = dyna_state

        dyna_runner_state, (losses, trajectories) = model_free_update_fn(
            dyna_runner_state,
            None,
        )

        dyna_state = DynaState(
            dyna_runner_state,
            model_train_state,
            model_env_state,
            rp_buf,
        )
        infos = (
            losses,
            trajectories,
            None,
            None,
            None,
        )
        return dyna_state, infos

    def _dyna_step(dyna_state: DynaState, _) -> Tuple[DynaState, Tuple]:
        # Note: State Passing is Horrible. Abstractions are bad.
        # Would do a rewrite but worried about break.
        dyna_runner_state, model_train_state, model_env_state, rp_buf = dyna_state

        rng = dyna_runner_state.rng
        rng, rng_model, rng_next = jax.random.split(rng, 3)
        #######
        dyna_runner_state, (losses, trajectories) = model_free_update_fn(
            dyna_runner_state,
            None,
        )
        sas_tup = trajectories.to_sas_tuple()
        rp_buf = rp_buf.insert(sas_tup)

        ######
        model_train_state, model_losses = env_model_update_fn(
            rng_model, model_train_state, trajectories
        )
        #######
        dyna_runner_state = DynaRunnerState(
            model_params=model_train_state.params,
            train_state=dyna_runner_state.get_train_state(),
            cartpole_env_state=model_env_state,
            last_obs=dyna_runner_state.last_obs,
            rng=rng_next,
        )

        #######
        dyna_runner_state, (p_loss, p_trajectories) = model_based_update_fn(
            dyna_runner_state,
            None,
        )
        final_runner_state = DynaRunnerState(
            model_params=model_train_state.params,
            train_state=dyna_runner_state.train_state,
            cartpole_env_state=dyna_runner_state.cartpole_env_state,
            last_obs=dyna_runner_state.last_obs,
            rng=rng_next,
        )

        final_dyna_state = DynaState(
            final_runner_state,
            model_train_state,
            dyna_runner_state.get_env_state(),
        )
        infos = (
            losses,
            trajectories,
            p_loss,
            p_trajectories,
            model_losses,
        )
        return final_dyna_state, infos

    if use_model:
        return _dyna_step
    else:
        return _model_free_step
