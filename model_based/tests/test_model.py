import jax
import jax.numpy as jnp

from model_based.nn_model import NNCartpole
from model_based.train import EquiModel, Model


def test_nn_model():
    key = jax.random.PRNGKey(42)
    env = NNCartpole()
    env_params = env.default_params
    num = 100
    env = NNCartpole(Model)

    m_param = env.transition_model.init(
        key,
        jnp.ones(env.observation_space(env_params).shape),
        jnp.ones(env.action_space().shape),
    )

    key = jax.random.split(key, num)
    actions = jax.random.randint(key[0], (num,), minval=0, maxval=2)
    _, base_env_state = jax.vmap(env.reset, in_axes=(0, None))(key, env_params)

    jax.vmap(env.step, in_axes=(0, 0, 0, None, None))(
        key,
        base_env_state,
        actions,
        env_params,
        m_param,
    )


def test_equi_model():
    key = jax.random.PRNGKey(42)

    env = NNCartpole(EquiModel)  # type: ignore
    assert_equiv_of(env.transition_model)
    env_params = env.default_params
    m_param = env.transition_model.init(
        key,
        jnp.ones(env.observation_space(env_params).shape),  # type: ignore
        jnp.ones(env.action_space().shape),
    )
    num = 100
    actions = jax.random.randint(key, (num,), minval=0, maxval=2)
    key = jax.random.split(key, num)
    _, base_env_state = jax.vmap(env.reset, in_axes=(0, None))(key, env_params)

    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None, None)
    )(
        key,
        base_env_state,
        actions,
        env_params,
        m_param,
    )
    neg_obsv, neg_env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None, None)
    )(
        key,
        jax.tree_map(lambda x: -x, base_env_state),
        1 - actions,
        env_params,
        m_param,
    )
    not_reset = obsv.at[done == 0].get()
    inv_not_reset = neg_obsv.at[done == 0].get()
    assert (not_reset == -inv_not_reset).all()


def assert_equiv_of(model):
    num = 100
    key = jax.random.PRNGKey(42)
    model_params = model.init(jax.random.PRNGKey(0), jnp.ones((4,)), jnp.ones((1,)))
    obs = jax.random.normal(
        key,
        (num, 4),
    )
    action = jax.random.randint(key, (100,), minval=0, maxval=2)
    obs_ = jax.vmap(model.apply, in_axes=(None, 0, 0))(model_params, obs, action)
    inv = jax.vmap(model.apply, in_axes=(None, 0, 0))(model_params, -obs, 1 - action)
    assert (obs_ == -inv).all()


def test_model():
    assert_equiv_of(EquiModel(4, 1, 10))


if __name__ == "__main__":
    test_equi_model()
    print("All tests passed!")
