import jax

from dyna.training import DynaHyperParams, make_dyna_train_fn

if __name__ == "__main__":
    hyp = DynaHyperParams()
    dyna_train_fn = make_dyna_train_fn(hyp)
    rng = jax.random.PRNGKey(42)
    with jax.disable_jit(False):
        dyna_train_fn(rng)
