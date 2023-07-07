from copy import deepcopy

import jax
import jax.numpy as jnp
import jaxtyping
from meta_rl.mutli_seed_script import CONFIG, ConvActorCritic, make_train

from model_based.nn_model import NNCartpole

if __name__ == "__main__":
    config = deepcopy(CONFIG)
    env = NNCartpole()
    config["ENV"] = env
    config["ENV_PARAMS"] = env.default_params()
    train_function = jax.vmap(jax.jit(make_train(config, ConvActorCritic)))
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 256)
    results = train_function(keys)
    episodic_returns = results["metrics"]["returned_episode_returns"].reshape((256, -1))
    jnp.save("NNCartpole.npy", episodic_returns)
