from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import numpy as np
import jax.numpy as jnp
import jaxtyping as jt
import optax
from base_rl.higher_order import FlattenObservationWrapper, Transition
from base_rl.models import ActorCritic
from base_rl.wrappers import LogWrapper
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray

from dyna.ac_higher_order import make_actor_critic_update
from dyna.model_higher_order import make_transition_model_update
from dyna.types import DynaHyperParams, DynaRunnerState, DynaState
from model_based.nn_model import NNModel


def make_dyna_train_fn(
    dyna_hyp: DynaHyperParams,
    nn_model: NNModel,
):
    tx_ac = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.ac_hyp.LR, eps=1e-5),
    )
    tx_model = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.model_hyp.LR, eps=1e-5),
    )
    env_model = LogWrapper(
        FlattenObservationWrapper(nn_model(model=dyna_hyp.model_hyp.MODEL_FN))
    )
    env = LogWrapper(FlattenObservationWrapper(env_model._env._env.parent_class()))
    actor_critic = ActorCritic(env.action_space(env.default_params).n)
    state_shape = np.prod(
        env.observation_space(env.default_params).shape, dtype=np.int32
    )
    action_shape = np.prod(env.action_space().shape, dtype=np.int32)

    def model_train_state_create(
        rng: jt.PRNGKeyArray,
    ) -> TrainState:
        if dyna_hyp.model_hyp.PARAMS:
            assert (
                dyna_hyp.model_hyp.NUM_EPOCHS == 0
            ), "Can't Use Expert Model with training."
            return TrainState.create(
                apply_fn=jax.vmap(
                    env_model.transition_model.apply, in_axes=(None, 0, 0)
                ),
                params=dyna_hyp.model_hyp.PARAMS,
                tx=tx_model,
            )

        else:
            return TrainState.create(
                apply_fn=jax.vmap(
                    env_model.transition_model.apply, in_axes=(None, 0, 0)
                ),
                params=env_model.transition_model.init(
                    rng, jnp.ones(state_shape), jnp.ones(action_shape)
                ),
                tx=tx_model,
            )

    def ac_train_state_create(rng: jt.PRNGKeyArray) -> TrainState:
        return TrainState.create(
            apply_fn=jax.vmap(actor_critic.apply, in_axes=(None, 0)),
            params=actor_critic.init(rng, jnp.ones(state_shape)),
            tx=tx_ac,
        )

    assert env.action_space().n == env_model.action_space().n
    assert (
        env.observation_space(env.default_params).shape
        == env_model.observation_space(env_model.default_params).shape
    )

    def train(rng):
        rng, rng_ac, rng_reset, rng_model, rng_state = jax.random.split(rng, 5)
        ac_train_state = ac_train_state_create(rng_ac)

        model_train_state = model_train_state_create(rng_model)
        # print(
        #     "Model Params:",
        #     sum(x.size for x in jax.tree_util.tree_leaves(model_train_state.params)),
        # )

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
        [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple]
    ],
    model_based_update_fn: Callable[
        [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple]
    ],
    env_model_update_fn: Callable[
        [PRNGKeyArray, TrainState, Transition], Tuple[TrainState, Tuple]
    ],
) -> Callable[[DynaState, Any], Tuple[DynaState, Tuple]]:
    def _model_free_step(dyna_state: DynaState, _) -> Tuple[DynaState, Tuple]:
        dyna_runner_state, model_train_state, model_env_state = dyna_state

        dyna_runner_state, (losses, trajectories) = model_free_update_fn(
            dyna_runner_state,
            None,
        )

        dyna_state = DynaState(
            dyna_runner_state,
            model_train_state,
            model_env_state,
        )
        infos = (
            losses,
            trajectories,
            None,
            None,
            None,
        )
        return dyna_state, infos

    def _experience_step(dyna_state: DynaState, _) -> Tuple[DynaState, Tuple]:
        # Note: State Passing is Horrible. Abstractions are bad.
        # Would do a rewrite but worried about break.
        dyna_runner_state, model_train_state, model_env_state = dyna_state

        rng = dyna_runner_state.rng
        rng, rng_model, rng_next = jax.random.split(rng, 3)
        #######
        dyna_runner_state, (losses, trajectories) = model_free_update_fn(
            dyna_runner_state,
            None,
        )
        final_obs = dyna_runner_state.last_obs
        final_env_state = dyna_runner_state.cartpole_env_state

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
            cartpole_env_state=final_env_state,
            last_obs=final_obs,
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
        return _experience_step
    else:
        return _model_free_step


def make_gain_exp(
    dyna_hyp: DynaHyperParams,
    model_free_update_fn: Callable[
        [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple]
    ],
    model_based_update_fn: Callable[
        [DynaRunnerState, Any], Tuple[DynaRunnerState, Tuple]
    ],
    env_model_update_fn: Callable[
        [PRNGKeyArray, TrainState, Transition], Tuple[TrainState, Tuple]
    ],
) -> Callable[[DynaState, Any], Tuple[DynaState, Tuple]]:
    def exp_step(
        dyna_state: DynaState, prev_traj: Transition
    ) -> Tuple[DynaState, Tuple]:
        dyna_runner_state, model_train_state, model_env_state = dyna_state

        rng = dyna_runner_state.rng
        rng, rng_model, rng_next = jax.random.split(rng, 3)
        #######
        dyna_runner_state, (losses, trajectories) = model_free_update_fn(
            dyna_runner_state,
            None,
        )
        trajectories = prev_traj.join(trajectories)
        final_env_state = dyna_runner_state.cartpole_env_state
        final_obs = dyna_runner_state.last_obs

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
            cartpole_env_state=final_env_state,
            last_obs=final_obs,
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

    @jax.jit
    def gain_exp(
        dyna_state: DynaState, prev_traj: Transition
    ) -> Tuple[DynaState, Tuple]:
        info = None
        for _ in range(dyna_hyp.NUM_UPDATES):
            dyna_state, new_info = exp_step(dyna_state, prev_traj)
            prev_traj = new_info[1]
            if info:
                info = jax.tree_map(
                    lambda x, y: jnp.concatenate([x, y]), info, new_info
                )
            else:
                info = new_info

        return (dyna_state, info)  # type: ignore

    return gain_exp


def stack_tuple_list(tup_list):
    items = [[] for _ in tup_list[0]]
    for tup in tup_list:
        for i, item in enumerate(tup):
            items[i].append(item)
    return tuple([jnp.stack(item) for item in items])
