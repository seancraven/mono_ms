import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, w=10000):
    return np.convolve(x, np.ones(w), "valid") / w


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # print("Param Count:", jax.tree_util.tree_map(lambda x: x.shape, params))
    returns = [np.load("./world_model_val.npy"), np.load("./world_model_train.npy")]
    for ret, lab in zip(returns, ["Acting", "Planning"]):
        episodic_returns = ret
        returns_std = moving_average(episodic_returns.std(axis=0)) / np.sqrt(256)
        returns_mean = moving_average(episodic_returns.mean(axis=0))
        returns_upper = returns_mean + returns_std
        returns_lower = returns_mean - returns_std
        x = np.arange(len(returns_mean))

        ax[0].plot(
            x,
            returns_mean,
            label=lab,
        )
        ax[0].fill_between(
            x,
            returns_lower,
            returns_upper,
            alpha=0.3,
        )
        ax[0].set_xlabel("Timesteps")
        ax[0].set_ylabel("Episodic Return")

        cumulative_returns = np.cumsum(episodic_returns, axis=1)[:, -1]
        assert cumulative_returns.shape == (256,), f"{cumulative_returns.shape}"
        worst_decile = np.quantile(cumulative_returns, 0.1)
        worst_mean = episodic_returns[cumulative_returns < worst_decile].mean(axis=0)
        worst_std = episodic_returns[cumulative_returns < worst_decile].std(
            axis=0
        ) / np.sqrt(256)
        worst_upper = worst_mean + worst_std
        worst_lower = worst_mean - worst_std
        ax[1].plot(
            x,
            moving_average(worst_mean),
            linestyle="--",
            label=lab,
        )
        ax[1].fill_between(
            x,
            moving_average(worst_lower),
            moving_average(worst_upper),
            alpha=0.3,
        )
    ax[1].set_xlabel("Timesteps")
    ax[1].set_ylabel("Worst Decile Episodic Return")
    ax[1].legend()

    fig.savefig("plots.png")
    plt.show()
