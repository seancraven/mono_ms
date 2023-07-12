from copy import deepcopy
from typing import Any, Callable, NamedTuple, Tuple, Union

import gymnax
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from distrax import Categorical
from flax.training.train_state import TrainState
from gymnax.environments.classic_control import CartPole
from meta_rl.models import ActorCritic
from meta_rl.mutli_seed_script import Obs, Params, RunnerState, Trajectory
from model_based.nn_model import NNCartpole, NNCartpoleParams

from dyna.ac_higher_order import (
    Losses,
    make_ac_mini_batch_update_fn,
    make_env_step_fn,
    make_gae_fn,
)

# CONFIG = {
#     "LR": 2.5e-4,
#     "NUM_ENVS": 4,
#     "NUM_STEPS": 128,
#     "TOTAL_TIMESTEPS": 5e5,
#     "UPDATE_EPOCHS": 4,
#     "NUM_MINIBATCHES": 4,
#     "GAMMA": 0.99,
#     "GAE_LAMBDA": 0.95,
#     "CLIP_EPS": 0.2,
#     "ENT_COEF": 0.01,
#     "VF_COEF": 0.5,
#     "MAX_GRAD_NORM": 0.5,
#     "ACTIVATION": "tanh",
#     "ANNEAL_LR": True,
#     "ENV": CartPole(),
#     "ENV_PARAMS": CartPole().default_params,
# }


class DynaRunnerState(NamedTuple):
    """Runner state for the dyna algorithm."""

    cartpole_env_state: gymnax.EnvState
    model_env_params: NNCartpoleParams
    train_state: TrainState
    last_obs: Obs
    rng: jt.PRNGKeyArray

    @property
    def cartpole_runner_state(self) -> RunnerState:
        # Cheap
        return deepcopy(
            (
                self.train_state,
                self.cartpole_env_state,
                self.last_obs,
                self.rng,
            )
        )


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
    pass


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


def make_dyna_train_fn(dyna_hyp: DynaHyperParams):
    actor_critic = ActorCritic(2)

    tx = optax.chain(
        optax.clip_by_global_norm(dyna_hyp.MAX_GRAD_NORM),
        optax.adam(dyna_hyp.ac_hyp.LR, eps=1e-5),
    )

    def train(rng):
        rng, rng_ac, rng_reset, rng_model, rng_state = jax.random.split(rng, 5)
        ac_train_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic.init(rng_ac, jnp.ones((1, 4))),
            tx=tx,
        )

        model_free_update_function = make_actor_critic_update(
            dyna_hyp,
            CartPole,
            actor_critic.apply,  # type: ignore
        )
        model_based_update_function = make_actor_critic_update(
            dyna_hyp,
            NNCartpole,
            actor_critic.apply,  # type: ignore
        )

        inital_dyna_runner_state = DynaRunnerState(
            cartpole_env_state=CartPole().reset(rng_reset),
            model_env_params=NNCartpole().TransitionModel.init(  # type: ignore
                rng_model, (jnp.ones((1, 4)), jnp.ones((1, 1)))
            ),
            train_state=ac_train_state,
            last_obs=jnp.zeros((1, 4)),
            rng=rng_state,
        )

        def _experience_step(dyna_runner_state: DynaRunnerState, _):
            cartpole_runner_state = dyna_runner_state.cartpole_runner_state

            cartpole_runner_state, (losses, trajectories) = model_free_update_function(
                cartpole_runner_state, None
            )
            # Extract the AC state from the cartpole runner state.

            # TRAIN MODEL
            # TODO: Train Model.
            model_runner_state, (
                planned_losses,
                planned_trajectories,
            ) = model_based_update_function(model_runner_state, None)

        final_dyna_runner, infos = jax.lax.scan(
            _experience_step, inital_dyna_runner_state, None
        )
        return final_dyna_runner, infos


def make_actor_critic_update(
    dyna_hyp: DynaHyperParams,
    env_creation_function: Callable[[], gymnax.environments.CartPole],
    model_apply_fn: Callable[[Params, jt.Array], Tuple[Categorical, Obs]],
) -> Callable[[RunnerState, Any], Tuple[RunnerState, Tuple[Losses, Trajectory]]]:
    # Throught training constant functions.
    env = env_creation_function()
    _mini_batch_update_fn = make_ac_mini_batch_update_fn(
        model_apply_fn, dyna_hyp.ac_hyp
    )
    _gae_fn = make_gae_fn(dyna_hyp)

    def actor_critic_update(
        runner_state: RunnerState, env_params: Union[gymnax.EnvParams, NNCartpoleParams]
    ) -> Tuple[RunnerState, Tuple[Losses, Trajectory]]:
        """Upates the train state of the actor critic model, on a given model.

        Notes: Model Is frozen don't need to update.
        Notes: Must Return trajectory.
        """
        # Changes depending on learned transition dynmaics.
        _env_step_fn = make_env_step_fn(dyna_hyp, env, env_params)

        def _model_free_update(runner_state: RunnerState, _):
            intermediate_runner_state, trajectories = jax.lax.scan(
                _env_step_fn,
                runner_state,
                None,
                length=dyna_hyp.AC_NUM_TRANSITIONS,
            )
            train_state, final_env_state, last_obs, rng = intermediate_runner_state
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

            final_runner_state: RunnerState = (
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


def make_transition_model_train(hyper_params: HyperParams):
    data: ReplayBuffer = pickle.load(open("replay_buffer.pickle", "rb"))
    data = SARSDTuple(*jax.tree_map(lambda x: x.astype(jnp.float32), data))
    non_term_index = data.done == 0
    data = jax.tree_map(lambda x: x.at[non_term_index, ...].get(), data)

    train_size = hyper_params.get_train_size(data)
    flattened_data = jax.tree_map(lambda x: x.reshape(-1, *x.shape[1:]), data)
    train_data, val_data = flattened_data.partition(train_size)
    val_data = jax.tree_map(lambda x: expand_scalar(x), val_data)

    def train(rng, train_data, val_data):
        rng = jax.random.PRNGKey(42)
        train_size = hyper_params.get_train_size(data)
        batch_count = hyper_params.get_batch_count(data)
        state_dim = data.state.shape[-1]
        action_dim = 1
        network = Model(state_dim, action_dim, hyper_params.hidden_dim)

        optimizer = adam(hyper_params.learning_rate)

        _, params_key = jax.random.split(rng)
        params = network.init(
            params_key, jnp.ones((state_dim,)), jnp.ones((action_dim,))
        )
        apply_network = jax.vmap(network.apply, in_axes=(None, 0, 0), out_axes=0)

        train_state = TrainState.create(
            apply_fn=apply_network,
            params=params,
            tx=optimizer,
        )

        logger.debug(f"state.shape: {data.state.shape}")
        logger.debug(f"state_dim: {state_dim}")
        logger.debug(f"action_dim: {action_dim}")

        def _epoch(
            joint_state: Tuple[jt.PRNGKeyArray, TrainState, SARSDTuple, SARSDTuple], _
        ) -> Tuple[
            Tuple[jt.PRNGKeyArray, TrainState, SARSDTuple, SARSDTuple],
            Tuple[LossData, DebugData, DebugData],
        ]:
            def _loss_fn(
                params: jt.PyTree, sarsd_tuple: SARSDTuple
            ) -> Tuple[jt.Array, DebugData]:
                state, action, _, next_state, _ = sarsd_tuple
                next_state_pred = train_state.apply_fn(params, state, action)

                next_state_loss = jnp.mean((next_state - next_state_pred) ** 2, axis=0)

                debug_loss = DebugData(
                    next_state_loss[0],
                    next_state_loss[1],
                    next_state_loss[2],
                    next_state_loss[3],
                )

                # reward_loss = jnp.mean(bce_from_logit(reward_pred_logit, reward))
                # done_loss = jnp.mean(bce_from_logit(done_pred_logit, done))

                return next_state_loss.mean(), debug_loss  # + reward_loss + done_loss

            def _mini_batch(
                train_state: TrainState, mini_batch: Any
            ) -> Tuple[TrainState, Tuple[jt.Array, DebugData]]:
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                (train_loss, debug_data), grad = grad_fn(train_state.params, mini_batch)
                train_state = train_state.apply_gradients(grads=grad)

                return train_state, (train_loss, debug_data)

            rng, train_state, train_data, val_data = joint_state
            _, rng = jax.random.split(rng)

            indecies = jax.random.permutation(rng, train_size)

            shuffle_train_data = jax.tree_map(
                lambda x: (x.at[indecies].get()).reshape(
                    (batch_count, hyper_params.batch_size, -1)
                ),
                train_data,
            )

            (train_state, (train_loss, debug_data)) = jax.lax.scan(
                _mini_batch,
                train_state,
                shuffle_train_data,
            )
            val_loss, val_debug_data = _loss_fn(train_state.params, val_data)

            return (rng, train_state, train_data, val_data), (
                LossData(train_loss, val_loss),
                debug_data,
                val_debug_data,
            )

        final_state, losses = jax.lax.scan(
            _epoch, (rng, train_state, train_data, val_data), None, hyper_params.epochs
        )

        return final_state, losses

    return lambda x: train(x, train_data, val_data)
