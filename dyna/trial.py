import jax
import matplotlib.pyplot as plt
from model_based.nn_model import Model

from dyna.ac_higher_order import ActorCriticHyperParams, DynaHyperParams
from dyna.training import make_dyna_train_fn
from dyna.types import TransitionModelHyperParams

# "NUM_ENVS": 4,
# "NUM_STEPS": 128,
# "TOTAL_TIMESTEPS": 5e5,
# "UPDATE_EPOCHS": 4,
if __name__ == "__main__":
    ac_hyp = ActorCriticHyperParams(
        NUM_EPOCHS=4,
        NUM_UPDATES=100,
    )
    tm_hyp = TransitionModelHyperParams(MODEL_FN=Model)
    hyp = DynaHyperParams(ac_hyp=ac_hyp, NUM_ENVS=4, NUM_UPDATES=10)
    rng = jax.random.PRNGKey(42)
    dyna_train_fn = make_dyna_train_fn(hyp)
    with jax.disable_jit():
        running_state, loss_info = dyna_train_fn(rng)
    mf_loss, mf_traj, p_loss, p_traj, m_loss = loss_info
    # plt.plot(mf_traj.info["returned_episode_returns"].mean(axis=-1).reshape(-1))
    # plt.show()
    plt.plot(mf_traj.info["returned_episode_returns"].mean(axis=-1).reshape(-1))
    plt.show()
    # plt.plot(mf_loss[0].reshape(-1))
    # plt.show()
