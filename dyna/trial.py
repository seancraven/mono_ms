import jax
import matplotlib.pyplot as plt

from dyna.training import DynaHyperParams, make_dyna_train_fn

if __name__ == "__main__":
    hyp = DynaHyperParams()
    dyna_train_fn = make_dyna_train_fn(hyp)
    rng = jax.random.PRNGKey(42)
    with jax.disable_jit(False):
        running_state, loss_info = dyna_train_fn(rng)
    mf_loss, mf_traj, p_loss, p_traj, m_loss = loss_info
    plt.plot(mf_traj.info["returned_episode_returns"].mean(axis=-1).reshape(-1))
    plt.show()
