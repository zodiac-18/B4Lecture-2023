"""Load a pickle file and predict HMM."""
import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_pickle(path):
    """
    Load pickle files.

    Args:
        path (str): A file path.

    Returns:
        ndarray: Output series.
        ndarray: Correct models with output series generated.
        ndarray: Initial state probability distribution.
        ndarray: State transition probability matrix.
        ndarray: Output probability.
    """
    data = pickle.load(open(path, "rb"))
    output = np.array(data["output"])
    answer_models = np.array(data["answer_models"])
    init_dist = np.array(data["models"]["PI"])
    trans_prob = np.array(data["models"]["A"])
    out_prob = np.array(data["models"]["B"])

    return output, answer_models, init_dist, trans_prob, out_prob


def forward_algorithm(output, init_dist, trans_prob, out_prob):
    """
    Predict HMM by forward algorithm.

    Args:
        output (ndarray): Output series.
        init_dist (ndarray): Initial state probability distribution.
        trans_prob (ndarray): State transition probability matrix.
        out_prob (ndarray): Output probability.

    Returns:
        ndarray: Predicted Models.
    """
    predict = np.zeros(output.shape[0])
    # calculate probabilities that series are generated
    for i in range(output.shape[0]):
        alpha = init_dist[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(1, output.shape[1]):
            alpha = (
                np.sum(trans_prob * alpha[:, :, None], axis=1)
                * out_prob[:, :, output[i, j]]
            )
        # select the model with the highest probability of generation
        predict[i] = np.argmax(np.sum(alpha, axis=1))

    return predict


def viterbi_algorithm(output, init_dist, trans_prob, out_prob):
    """
    Predict HMM by viterbi algorithm.

    Args:
        output (ndarray): Output series.
        init_dist (ndarray): Initial state probability distribution.
        trans_prob (ndarray): State transition probability matrix.
        out_prob (ndarray): Output probability.

    Returns:
        ndarray: Predicted Models.
    """
    predict = np.zeros(output.shape[0])
    # calculate maximum probabilities
    for i in range(output.shape[0]):
        delta = init_dist[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(1, output.shape[1]):
            delta = (
                np.max(trans_prob * delta[:, :, None], axis=1)
                * out_prob[:, :, output[i, j]]
            )
        # select the model with the highest probability
        predict[i] = np.argmax(np.max(delta, axis=1))

    return predict


def cm_plot(predict, answer, algorithm, runtime):
    """
    Plot predicted results on a heat map.

    Args:
        predict (ndarray): Predicted Models.
        answer (ndarray): Correct models with output series generated.
        algorithm (str): Name of used algorithm.
        runtime (float): Runtime of used algorithm.

    Returns:
        None
    """
    # create confusion matrix
    number = np.unique(answer)
    cm = confusion_matrix(answer, predict)
    cm = pd.DataFrame(cm, columns=number, index=number)
    # calculate accuracy
    acc = np.sum(predict - answer == 0) / len(answer) * 100
    # plot confusion matrix
    sns.heatmap(cm, annot=True, cbar=False, cmap="Reds")
    plt.title(algorithm + f"\nAccuracy: {acc}%\nRumtime: {runtime:.4f} s")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")


def main():
    """
    Load pickle files and predict HMM.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="Demonstration of argparser",
        description="description",
        epilog="end",
        add_help=True,
    )
    # add arguments
    parser.add_argument("-f", dest="filename", help="Filename", required=True)
    # parse arguments
    args = parser.parse_args()
    path = args.filename

    # load pickle data
    output, answer_models, init_dist, trans_prob, out_prob = load_pickle(path)

    # execute algorithms and measure execution times
    forward_start = time.perf_counter()
    forward_pred = forward_algorithm(output, init_dist, trans_prob, out_prob)
    forward_stop = time.perf_counter()
    viterbi_start = time.perf_counter()
    viterbi_pred = viterbi_algorithm(output, init_dist, trans_prob, out_prob)
    viterbi_stop = time.perf_counter()
    # calculate measured execution times
    forward_time = forward_stop - forward_start
    viterbi_time = viterbi_stop - viterbi_start

    # display results
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.figure(figsize=(11, 6))
    plt.subplot(121)
    cm_plot(forward_pred, answer_models, "Forward algorithm", forward_time)
    plt.subplot(122)
    cm_plot(viterbi_pred, answer_models, "Viterbi algorithm", viterbi_time)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
