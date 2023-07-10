import pickle

import matplotlib.pyplot as plt
import numpy as np

from model_based.learn_trainsition_dynamics import DebugData


def debug_from_file(path: str) -> DebugData:
    debug_data: DebugData = pickle.load(open(path, "rb"))
    return debug_data


if __name__ == "__main__":
    train_loss_path = "./transition_model_train_loss.npy"
    val_loss_path = "./transition_model_val_loss.npy"
    train_debug = debug_from_file("./transition_model_debug_loss.pickle").flatten()
    val_debug = debug_from_file("./transition_model_val_debug_loss.pickle").flatten()
    train_loss = np.load(train_loss_path)
    val_loss = np.load(val_loss_path)
    ## Plot loss
    # plt.plot(train_loss, label="train_loss")
    # plt.plot(val_loss, label="val_loss")
    # plt.show()

    fig, ax = plt.subplots(2, 2)
    for i, value in enumerate(train_debug):
        ax[i // 2, i % 2].semilogy(
            value,
        )
    # for i, value in enumerate(val_debug):
    #     ax[i // 2, i % 2].plot(
    #         value,
    #     )
    plt.show()
