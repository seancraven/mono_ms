import jax
import jax.numpy as jnp

from model_based.nn_model import NNCartpole


def test_nn_model():
    key = jax.random.PRNGKey(0)
    env = NNCartpole()
    env_params = env.default_params()
    obs, env_state = env.reset(key, env_params)
    assert obs.shape == (4,)
    _, key = jax.random.split(key)
    actions = jnp.ones((100, 1))
    for action in actions:
        _, key = jax.random.split(key)
        obs, env_state, reward, done, info = env.step(
            key, env_state, action, env_params
        )


if __name__ == "__main__":
    test_nn_model()
    pass
