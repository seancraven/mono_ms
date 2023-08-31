from dyna.training import make_dyna_train_fn
from dyna.global_config import make_hyp_cp
from dyna.types import TransitionModelHyperParams
from model_based.nn_model import NNCartPole
from model_based.transition_models import Model, TransitionModel, EquiModel
import jax
import jaxtyping as jt
from typing import Callable, NamedTuple, Any
import pickle
import tqdm


class ResultTup(NamedTuple):
    tm_loss: Any
    returns: Any


def model_free_train(rng, num_seeds=128):
    rngs = jax.random.split(rng, num_seeds)
    tm = TransitionModelHyperParams(
        USE_MODEL=False,
        NUM_EPOCHS=0,
    )
    hyp = make_hyp_cp()
    hyp = hyp._replace(model_hyp=tm)
    model_free_tf = jax.jit(jax.vmap(make_dyna_train_fn(hyp, NNCartPole)))
    _, metrics = model_free_tf(rngs)
    return metrics


def memory_limited_train(
    rng: jt.PRNGKeyArray,
    pr: int,
    dyna_iter: int,
    transition_constructor: Callable[[], TransitionModel],
    num_seeds: int = 128,
):
    rngs = jax.random.split(rng, num_seeds)
    rngs = rngs.reshape((8, -1, 2))
    jax.random.split(rng, num_seeds)
    hyp = make_hyp_cp(num_dyna_itr=dyna_iter)
    tm = TransitionModelHyperParams(
        MODEL_FN=transition_constructor,
    )
    hyp = hyp._replace(model_hyp=tm, PLANNING_RATIO=pr)
    train_fn = jax.jit(jax.vmap(make_dyna_train_fn(hyp, NNCartPole)))

    def scannable_train(_, rng):
        _, metrics = train_fn(rng)
        tm_losses = metrics[-1]
        returns = metrics[1].info["returned_episode_returns"]
        return None, ResultTup(tm_losses, returns)

    metrics = jax.lax.scan(
        scannable_train,
        None,
        rngs,
    )
    return metrics


planning_ratios = [0.5, 1, 2, 4, 8]
dyna_iters = [1, 5, 10, 50, 100]

if __name__ == "__main__":
    cpu = jax.devices("cpu")[0]
    rng = jax.random.PRNGKey(42)
    # print("baseline")
    # model_free_results = model_free_train(rng)
    # with open("./base_metrics.pkl", "wb") as f_base:
    #     pickle.dump(model_free_results, f_base)
    # del model_free_results
    for pr in tqdm.tqdm(planning_ratios):
        for iter_ in dyna_iters:
            print(f"PR: {pr} Iter: {iter_}")
            model_sweep_results = memory_limited_train(rng, pr, iter_, Model)  # type: ignore
            with open(f"./dyna_pr{pr}_iter{iter_}.pkl", "wb") as f:
                pickle.dump(jax.device_put(model_sweep_results, cpu), f)
            del model_sweep_results

    print("The Equi boi")
    for pr in tqdm.tqdm(planning_ratios):
        for iter_ in dyna_iters:
            print(f"PR: {pr} Iter: {iter_}")
            model_sweep_results = memory_limited_train(rng, pr, iter_, EquiModel)  # type: ignore
            with open(f"./equi_dyna_pr{pr}_iter{iter_}.pkl", "wb") as equi_f:
                pickle.dump(jax.device_put(model_sweep_results, cpu), equi_f)
            del model_sweep_results
