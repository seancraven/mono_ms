from typing import Any, Callable, Dict, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from base_rl.higher_order import (
    BatchData,
    CartPoleRunnerState,
    Params,
    PerTimestepScalar,
    Trajectory,
    Transition,
)
from base_rl.models import ActorCritic
from base_rl.wrappers import FlattenObservationWrapper, LogWrapper
from flax.training.train_state import TrainState
from gymnax.environments.classic_control import CartPole

from model_based.nn_model import NNCartpole

UpdateState = Tuple[
    TrainState,
    Tuple[Transition, Transition],
    PerTimestepScalar,
    PerTimestepScalar,
    jt.PRNGKeyArray,
]


def make_train(
    config: Dict, modle_creation_fn: Callable[[int], nn.Module]
) -> Callable[[jt.PRNGKeyArray], Dict[str, Union[TrainState, Any]]]:
    """Make training fuction with evaluation on a validaiton environment."""
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = config["ENV"]
    env_params = config["ENV_PARAMS"]
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)  # type: ignore
    network = modle_creation_fn(4)

    if "VAL_ENV" in config:
        env_val = config["VAL_ENV"]
        env_val_params = config["VAL_ENV_PARAMS"]
        env_val = FlattenObservationWrapper(env_val)
        env_val = LogWrapper(env_val)  # type: ignore

    else:
        raise NotImplementedError("Use other function!")

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng: jt.PRNGKeyArray) -> Dict[str, Union[TrainState, Any]]:
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, 4))
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state: CartPoleRunnerState, _):
            """Trajectories are evaluated and performs Adam on batched trajectories."""

            def _val_env_step(
                val_runner_state: CartPoleRunnerState, _
            ) -> Tuple[CartPoleRunnerState, Transition]:
                train_state, val_env_state, last_obs, rng = val_runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, val_env_state, reward, done, info = jax.vmap(
                    env_val.step, in_axes=(0, 0, 0, None)
                )(rng_step, val_env_state, action, env_val_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info  # type: ignore
                )
                runner_state = (train_state, val_env_state, obsv, rng)  # type: ignore
                return runner_state, transition

            def _env_step(
                runner_state: CartPoleRunnerState, _
            ) -> Tuple[CartPoleRunnerState, Transition]:
                train_state, env_state, last_obs, rng = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info  # type: ignore
                )
                runner_state = (train_state, env_state, obsv, rng)  # type: ignore
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            _, val_traj_batch = jax.lax.scan(
                _val_env_step, runner_state, None, config["NUM_STEPS"]
            )
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(
                traj_batch: Trajectory,
                last_val: jt.Array,
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
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,  # type: ignore
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)  # type: ignore

            # UPDATE NETWORK
            def _update_epoch(update_state: UpdateState, _):
                """Loss evaluated and Adam applied across all minibatches."""

                def _update_minbatch(
                    train_state: TrainState, batch_info: BatchData
                ) -> Tuple[
                    TrainState, Tuple[jt.Array, Tuple[jt.Array, jt.Array, jt.Array]]
                ]:
                    """Loss evaluated and Adam applied to a single minibatch."""
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(
                        params: Params,
                        traj_batch: Trajectory,
                        gae: PerTimestepScalar,
                        targets: PerTimestepScalar,
                    ) -> Tuple[jt.Array, Tuple[jt.Array, jt.Array, jt.Array]]:
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, train_val_traj, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                train_val_traj = (traj_batch, val_traj_batch)
                update_state = (train_state, train_val_traj, advantages, targets, rng)
                return update_state, total_loss

            train_val_traj = (traj_batch, val_traj_batch)
            update_state = (train_state, train_val_traj, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]  # type: ignore
            )
            train_state = update_state[0]
            metric = (train_val_traj[0].info, train_val_traj[1].info)
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


CONFIG = {
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "ENV": NNCartpole(),
    "ENV_PARAMS": NNCartpole().default_params,
    "VAL_ENV": CartPole(),
    "VAL_ENV_PARAMS": CartPole().default_params,
}

if __name__ == "__main__":
    num_seeds = 256
    key = jax.random.PRNGKey(42)
    sym_key, key = jax.random.split(key)

    model = ActorCritic

    keys = jax.random.split(key, num_seeds)

    train_fn = jax.vmap(jax.jit(make_train(CONFIG, model)))
    return_dict = train_fn(keys)
    metrics = return_dict["metrics"]
    results_train = metrics[0]
    results_val = metrics[1]
    episodic_returns_val = results_val["returned_episode_returns"].reshape(
        (num_seeds, -1)
    )
    episodic_returns_train = results_train["returned_episode_returns"].reshape(
        (num_seeds, -1)
    )
    jnp.save("world_model_train.npy", episodic_returns_train)
    jnp.save("world_model_val.npy", episodic_returns_val)
