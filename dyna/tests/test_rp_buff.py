import jax
import jax.numpy as jnp

from dyna.types import ReplayBuffer, SASTuple


def test_rp_buff():
    key = jax.random.PRNGKey(42)
    dummy_sas = SASTuple(
        state=jnp.ones((4,)),
        action=jnp.ones((1)),
        next_state=jnp.ones((4)),
    )
    rp_buff = ReplayBuffer.create(key, dummy_sas, 10)
    assert rp_buff.data.state.shape == (10, 4)


def test_rp_buff_insert():
    key = jax.random.PRNGKey(42)
    dummy_sas = SASTuple(
        state=jnp.ones((4)),
        action=jnp.ones((1)),
        next_state=jnp.ones((4)),
    )
    rp_buff = ReplayBuffer.create(key, dummy_sas, 10)

    states = jnp.empty((2, 4))
    actions = jnp.empty((2, 1))
    next_states = jnp.empty((2, 4))
    sas = SASTuple(state=states, action=actions, next_state=next_states)
    rp_buff = rp_buff.insert(sas)
    assert (rp_buff.data.state.at[0:2].get() == states).all()
    assert (rp_buff.data.action.at[0:2].get() == actions).all()
    assert (rp_buff.data.next_state.at[0:2].get() == next_states).all()


def test_rp_buff_sample():
    key = jax.random.PRNGKey(42)
    dummy_sas = SASTuple(
        state=jnp.ones((4)),
        action=jnp.ones((1)),
        next_state=jnp.ones((4)),
    )
    rp_buff = ReplayBuffer.create(key, dummy_sas, 10)

    states = jnp.ones((2, 4)) * 6
    actions = jnp.ones((2, 1)) * 7
    next_states = jnp.ones((2, 4)) * 8
    sas = SASTuple(state=states, action=actions, next_state=next_states)
    rp_buff = rp_buff.insert(sas)

    buff, sample = rp_buff.sample()
    assert (sample.state == jnp.ones((10, 4)) * 6).all()
    assert (sample.next_state == jnp.ones((10, 4)) * 8).all()
    assert (sample.action == jnp.ones((10, 1)) * 7).all()


def test_rp_buff_queue():
    key = jax.random.PRNGKey(42)
    key = jax.random.PRNGKey(42)
    dummy_sas = SASTuple(
        state=jnp.ones((4)),
        action=jnp.ones((1)),
        next_state=jnp.ones((4)),
    )
    rp_buff = ReplayBuffer.create(key, dummy_sas, 10)

    for _ in range(5):
        states = jnp.ones((2, 4)) * 6
        actions = jnp.ones((2, 1)) * 7
        next_states = jnp.ones((2, 4)) * 8
        sas = SASTuple(state=states, action=actions, next_state=next_states)
        rp_buff = rp_buff.insert(sas)

    states = jnp.ones((2, 4)) * 10
    actions = jnp.ones((2, 1)) * 10
    next_states = jnp.ones((2, 4)) * 10

    sas = SASTuple(state=states, action=actions, next_state=next_states)
    rp_buff = rp_buff.insert(sas)
