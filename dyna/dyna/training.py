from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from flax.training.train_state import TrainState
from gymnax.environments.classic_control import CartPole
from meta_rl.models import ActorCritic
from meta_rl.mutli_seed_script import Obs, Params
from model_based.nn_model import NNCartpole

from dyna.ac_higher_order import make_actor_critic_update
from dyna.model_higher_order import make_transition_model_update


class ActorCriticHyperParams(NamedTuple):
    """Hyper parameters for the actor critic model."""

    NUM_UPDATES: int = 10
    NUM_EPOCHS: int = 10
    MINIBATCH_SIZE: int = 64
    _NUM_TIMESTEPS: int = 128

    CLIP_EPS: float = 0.2
    VF_COEF: float = 0.5
    ENT_COEF: float = 0.01
    LR: float = 2.5e-4


class TransitionModelHyperParams(NamedTuple):
    MINIBATCH_SIZE: int = 64
    NUM_EPOCHS: int = 10
    LR: float = 1e-3


class DynaHyperParams(NamedTuple):
    NUM_ENVS: int
    GAMMA: float
    GAE_LAMBDA: float

    MAX_GRAD_NORM: float

    ac_hyp: ActorCriticHyperParams
    model_hyp: TransitionModelHyperParams

    @property
    def AC_NUM_MINIBATCHES(self) -> int:
        return (self.NUM_ENVS * self.AC_NUM_TIMESTEPS) // self.ac_hyp.MINIBATCH_SIZE

    @property
    def AC_NUM_TRANSITIONS(self) -> int:
        return self.NUM_ENVS * self.ac_hyp.MINIBATCH_SIZE * self.AC_NUM_MINIBATCHES

    @property
    def AC_BATCH_SIZE(self) -> int:
        return self.NUM_ENVS * self.AC_NUM_TRANSITIONS

    @property
    def AC_NUM_TIMESTEPS(self) -> int:
        return self.AC_NUM_MINIBATCHES * self.ac_hyp.MINIBATCH_SIZE // self.NUM_ENVS

    @property
    def M_NUM_MINIBATCHES(self) -> int:
        assert self.AC_NUM_TRANSITIONS % self.model_hyp.MINIBATCH_SIZE == 0
        return self.AC_NUM_TRANSITIONS // self.model_hyp.MINIBATCH_SIZE


class DynaRunnerState(NamedTuple):
    model_params: Optional[Params]
    train_state: TrainState
    cartpole_env_state: gymnax.EnvState
    last_obs: Obs
    rng: jt.PRNGKeyArray


class DynaState(NamedTuple):
    dyna_runner_state: DynaRunnerState
    model_train_state: TrainState


def make_dyna_train_fn(dyna_hyp: DynaHyperParams):
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
        rng, rng_ac, rng_reset, rng_model, rng_state = jax.random.split(rng, 5)
        ac_train_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic.init(rng_ac, jnp.ones((1, 4))),
            tx=tx_ac,
        )
        model_train_state = TrainState.create(
            apply_fn=NNCartpole().transition_model.apply,
            params=NNCartpole().transition_model.init(  # type: ignore
                rng_model, (jnp.ones((1, 4)), jnp.ones((1, 1)))
            ),
            tx=tx_model,
        )

        model_free_update_fn = make_actor_critic_update(
            dyna_hyp,
            CartPole,
            ac_train_state.apply_fn,
            False,
        )
        model_based_update_fn = make_actor_critic_update(
            dyna_hyp,
            NNCartpole,
            model_train_state.apply_fn,
            True,
        )

        env_model_update_fn = make_transition_model_update(
            dyna_hyp,
            model_train_state.apply_fn,
        )

        dyna_runner_state = DynaRunnerState(
            cartpole_env_state=CartPole().reset(rng_reset),
            model_params=model_train_state.params,
            train_state=ac_train_state,
            last_obs=jnp.zeros((1, 4)),
            rng=rng_state,
        )

        def _experience_step(dyna_state: DynaState, _) -> Tuple[DynaState, Tuple]:
            dyna_runner_state, model_train_state = dyna_state
            dyna_runner_state, (losses, trajectories) = model_free_update_fn(
                dyna_runner_state, None
            )
            dyna_runner_state, (p_loss, p_trajectories) = model_based_update_fn(
                dyna_runner_state,
                None,
            )
            rng = dyna_runner_state.rng
            rng, rng_model = jax.random.split(rng)
            model_train_state, model_losses = env_model_update_fn(
                rng_model, model_train_state, trajectories
            )
            rng, rng_next = jax.random.split(rng)

            infos = (
                losses,
                trajectories,
                p_loss,
                p_trajectories,
                model_losses,
            )
            new_runner_state = DynaRunnerState(
                model_params=model_train_state.params,
                train_state=dyna_runner_state.train_state,
                cartpole_env_state=dyna_runner_state.cartpole_env_state,
                last_obs=dyna_runner_state.last_obs,
                rng=rng_next,
            )

            new_dyna_state = DynaState(new_runner_state, model_train_state)
            return new_dyna_state, infos

        final_dyna_runner, infos = jax.lax.scan(
            _experience_step,
            DynaState(dyna_runner_state, model_train_state),
            None,
        )
        return final_dyna_runner, infos

    return train
