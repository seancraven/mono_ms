import flax.linen as nn
import jax
import jax.numpy as jnp
from g_conv.c2 import C2Dense, C2DenseBinary

from model_based.nn_model import NNCartpole
from model_based.transition_models import EquiModel, Model


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


class Equi(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, state, action):
        action = jnp.array((action,))
        state_embedding = nn.tanh(C2Dense(self.hidden_dim)(state))
        action_embedding = nn.tanh(C2DenseBinary(self.hidden_dim)(action))

        # assert (
        #     state_embedding.shape == action_embedding.shape
        # ), f"{state_embedding.shape} != {action_embedding.shape}"

        concat = jnp.concatenate(
            [state_embedding.reshape(-1), action_embedding.reshape(-1)], axis=0
        )
        return concat


def test_equi():
    assert_equiv_of(Equi(2))


def proximal_state(state, next_states):
    assert next_states.shape[-1] == 2, f"{next_states.shape}"
    normal_state = next_states.at[..., 0].get()
    morror_state = next_states.at[..., 1].get()
    distance_next = ((normal_state.squeeze() - state.squeeze()) ** 2).sum()
    distance_mirror = ((morror_state.squeeze() - state.squeeze()) ** 2).sum()
    next_is_proximal = distance_next < distance_mirror
    return jax.lax.cond(next_is_proximal, lambda: normal_state, lambda: morror_state)


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
    num = 10
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
    num = 2
    key = jax.random.PRNGKey(42)
    model_params = model.init(jax.random.PRNGKey(0), jnp.ones((4,)), jnp.ones((1,)))
    obs = jax.random.normal(
        key,
        (num, 4),
    )
    action = jax.random.randint(key, (num, 1), minval=0, maxval=2)
    obs_ = jax.vmap(model.apply, in_axes=(None, 0, 0))(model_params, obs, action)
    inv = jax.vmap(model.apply, in_axes=(None, 0, 0))(model_params, -obs, 1 - action)
    print("Partial Solution: ", (obs_ == -inv).prod(axis=0).mean())
    print("Solution: ", (obs_ == -inv))
    assert (obs_ == -inv).all()


def test_model():
    assert_equiv_of(EquiModel(4, 1, 10))


if __name__ == "__main__":
    test_equi_model()
    print("All tests passed!")
