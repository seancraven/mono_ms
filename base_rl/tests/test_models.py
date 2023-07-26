import jax
import jax.numpy as jnp

from base_rl.models import EquivariantActorCritic


def test_equivaraint_ac():
    rng = jax.random.PRNGKey(0)
    model = EquivariantActorCritic(10)
    params = model.init(rng, jnp.ones((4,)))

    _, rng = jax.random.split(rng)
    state = jax.random.normal(rng, (10, 4))

    def dist_prob(x, i):
        pi, v = model.apply(params, x)
        return pi.log_prob(i)

    def v(x):
        pi, v = model.apply(params, x)
        return v

    get_vs = jax.vmap(
        v,
    )

    get_probs = jax.vmap(dist_prob, in_axes=(0, None))
    assert jnp.allclose(get_probs(state, 0), get_probs(-state, 1))
    assert jnp.allclose(get_vs(state), get_vs(-state))
