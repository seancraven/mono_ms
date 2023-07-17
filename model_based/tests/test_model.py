import jax
import jax.numpy as jnp

from model_based.nn_model import NNCartpole
from model_based.train import EquiModel


def test_nn_model():
    key = jax.random.PRNGKey(0)
    env = NNCartpole()
    m_param = env.transition_model.init(key, jnp.ones((4,)), jnp.ones((1,)))
    env_params = env.default_params
    obs, env_state = env.reset(key, env_params)
    assert obs.shape == (4,)
    _, key = jax.random.split(key)
    actions = jnp.ones((100, 1))
    for action in actions:
        _, key = jax.random.split(key)
        obs, env_state, reward, done, info = env.step(
            key,
            env_state,
            action,
            env_params,
            m_param,
        )


def test_equi_model():
    key = jax.random.PRNGKey(42)

    env = NNCartpole(EquiModel)  # type: ignore
    env_params = env.default_params
    m_param = env.transition_model.init(
        key,
        jnp.ones(env.observation_space(env_params).shape),  # type: ignore
        jnp.ones(env.action_space().shape),
    )
    _, env_state = env.reset(key, env_params)
    _, key = jax.random.split(key)
    actions = jnp.ones((100, 1))
    for action in actions:
        _, key = jax.random.split(key)
        _, env_state, _, _, _ = env.step(key, env_state, action, env_params, m_param)
        _, inv_env_state, _, _, _ = env.step(
            key, jax.tree_map(lambda x: -x, env_state), 0, env_params, m_param
        )
        assert jax.tree_map(
            lambda x, y: jnp.allclose(x, y),
            env_state,
            jax.tree_map(lambda x: -x, inv_env_state),
        )
