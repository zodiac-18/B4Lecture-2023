"""Predict HMM."""
import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as skl


def load_pickle(path):
    """
    Load data from pickle files.

    Args:
        path (str): The path of the data.

    Returns:
        tuple: Output, answer, initial probability, state transition probability, output probablity.
    """
    data = pickle.load(open(path, "rb"))
    output = np.array(data["output"])
    answer_models = np.array(data["answer_models"])
    init_prob = np.array(data["models"]["PI"])
    trans_prob = np.array(data["models"]["A"])
    out_prob = np.array(data["models"]["B"])

    return output, answer_models, init_prob, trans_prob, out_prob


class HMM:
    """Hidden Markov Model."""

    def __init__(self, output, answer_models, init_prob, trans_prob, out_prob):
        """
        Initialize instances.

        Args:
            output (ndarray): Output.
            answer_models (ndarray): Answer.
            init_prob (ndarray): Initial state probability distribution.
            trans_prob (ndarray): State transition probability.
            out_prob (ndarray): Output probability.
        """
        self.output = output
        self.answer_models = answer_models
        self.init_prob = init_prob
        self.trans_prob = trans_prob
        self.out_prob = out_prob

    def forward_algorithm(self):
        """
        Forward algorithm.

        Returns:
            tuple: Array of the probability of output and array of predicted output.
        """
        P, T = self.output.shape
        prob_gen, predict = np.zeros((P, self.init_prob.shape[0])), np.zeros(P)
        for i in range(P):
            alpha = self.init_prob[:, :, 0] * self.out_prob[:, :, self.output[i, 0]]
            for j in range(1, T):
                alpha = (
                    np.sum(alpha[:, :, np.newaxis] * self.trans_prob, axis=1)
                    * self.out_prob[:, :, self.output[i, j]]
                )
            prob_gen[i] = np.sum(alpha, axis=1)
            predict[i] = np.argmax(prob_gen[i])
        return prob_gen, predict

    def viterbi_algorithm(self):
        """
        Viterbi algorithm.

        Returns:
            tuple: Array of the probability of output and array of predicted output.
        """
        P, T = self.output.shape
        K = self.init_prob.shape[0]
        prob_gen, predict = np.zeros((P, K)), np.zeros(P)
        for i in range(P):
            delta = self.init_prob[:, :, 0] * self.out_prob[:, :, self.output[i, 0]]
            for j in range(1, T):
                delta = (
                    np.max(delta[:, :, np.newaxis] * self.trans_prob, axis=1)
                    * self.out_prob[:, :, self.output[i, j]]
                )
            prob_gen[i] = np.max(delta, axis=1)
            predict[i] = np.argmax(prob_gen[i])
        return prob_gen, predict

    def plot_confusion_matrix(self, predict, answer, runtime, algo_name, ax):
        """
        Plot confusion matrix.

        Args:
            predict (ndarray): Predicted output.
            answer (ndarray): Correct output.
            runtime (ndarray): Runtime of each algorithm.
            algo_name (str): The name of used algorithm.
            ax (matplotlib.axes._axes.Axes): Axes.
        """
        cm = skl.confusion_matrix(answer, predict)
        acc = skl.accuracy_score(answer, predict) * 100
        sns.heatmap(cm, annot=True, cbar=False, cmap="coolwarm")
        ax.set_title(algo_name + f"\nAccuracy: {acc}%\n(runtime: {runtime:.03f}s)")
        ax.set_xlabel("Predicted model")
        ax.set_ylabel("Actual model")
        return (
            f"------------Classification report of {algo_name}------------\n\n"
            + skl.classification_report(answer, predict)
        )


def main():
    """Predict HMM."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
    args = parser.parse_args()
    # Get the path of data
    path = args.path
    # Get the filename of data from the path
    file_name = os.path.splitext(os.path.basename(path))[0]

    output, answer_models, init_dist, trans_prob, out_prob = load_pickle(path)

    hmm = HMM(output, answer_models, init_dist, trans_prob, out_prob)
    # Run the forward algorithm.
    time_forward_start = time.perf_counter()
    _, forward_pred = hmm.forward_algorithm()
    time_forward_finish = time.perf_counter()

    forward_runtime = time_forward_finish - time_forward_start

    # Run the viterbi algorithm.
    time_viterbi_start = time.perf_counter()
    _, viterbi_pred = hmm.viterbi_algorithm()
    time_viterbi_finish = time.perf_counter()

    viterbi_runtime = time_viterbi_finish - time_viterbi_start

    report_str = ""

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"Confusion matrix of {file_name}")
    ax1 = fig.add_subplot(1, 2, 1)
    report_str += hmm.plot_confusion_matrix(
        forward_pred, answer_models, forward_runtime, "Forward algorithm", ax1
    )
    ax2 = fig.add_subplot(1, 2, 2)
    report_str += hmm.plot_confusion_matrix(
        viterbi_pred, answer_models, viterbi_runtime, "Viterbi algorithm", ax2
    )
    fig.savefig(f"con_mat_{file_name}.png")
    f = open(f"result_{file_name}.txt", "w")
    f.write(report_str)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
