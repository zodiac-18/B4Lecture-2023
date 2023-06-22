import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import argparse
import time


def load_pickle(path):
    data = pickle.load(open(path, "rb"))
    output = np.array(data["output"])
    answer_models = np.array(data["answer_models"])
    init_dist = np.array(data["models"]["PI"])
    trans_prob = np.array(data["models"]["A"])
    out_prob = np.array(data["models"]["B"])

    return output, answer_models, init_dist, trans_prob, out_prob


def forward_algorithm(output, init_dist, trans_prob, out_prob):
    predict = np.zeros(output.shape[0])
    # calculate probabilities that series are generated
    for i in range(output.shape[0]):
        alpha = init_dist[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(1, output.shape[1]):
            alpha = np.sum(trans_prob * alpha[:, :, None], axis=1) * out_prob[:, :, output[i, j]]
        # select the model with the highest probability of generation
        predict[i] = np.argmax(np.sum(alpha, axis=1))

    return predict


def viterbi_algorithm(output, init_dist, trans_prob, out_prob):
    predict = np.zeros(output.shape[0])
    # calculate maximum probabilities
    for i in range(output.shape[0]):
        delta = init_dist[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(1, output.shape[1]):
            delta = np.max(trans_prob * delta[:, :, None], axis=1) * out_prob[:, :, output[i, j]]
        # select the model with the highest probability
        predict[i] = np.argmax(np.max(delta, axis=1))

    return predict


def cm_plot(predict, answer, algorithm):
    # create confusion matrix
    number = np.unique(answer)
    cm = confusion_matrix(answer, predict)
    cm = pd.DataFrame(cm, columns=number, index=number)
    # calculate accuracy
    acc = np.sum(predict - answer == 0) / len(answer) * 100
    # plot confusion matrix
    sns.heatmap(cm, annot=True, cbar=False, cmap="Reds")
    plt.title(algorithm + f"\n(Accuracy : {acc}%)")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")


def main():
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

    forward_time = forward_stop - forward_start
    viterbi_time = viterbi_stop - viterbi_start

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    # display results
    print(f"Execution time of forward algorithm : {forward_time:.5f} [s]")
    print(f"Execution time of viterbi algorithm : {viterbi_time:.5f} [s]")
    plt.figure(figsize=(11, 5))
    plt.subplot(121)
    cm_plot(forward_pred, answer_models, "Forward algorithm")
    plt.subplot(122)
    cm_plot(viterbi_pred, answer_models, "Viterbi algorithm")
    plt.show()


if __name__ == "__main__":
    main()