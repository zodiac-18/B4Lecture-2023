import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import argparse


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
            alpha = np.sum(trans_prob * alpha[:, :, None],
                           axis=1) * out_prob[:, :, output[i, j]]
        # select the model with the highest probability of generation
        predict[i] = np.argmax(np.sum(alpha, axis=1))

        return predict


def viterbi_algorithm(output, init_dist, trans_prob, out_prob):
    predict = np.zeros(output.shape[0])
    # calculate maximum probabilities
    for i in range(output.shape[0]):
        delta = init_dist[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(1, output.shape[1]):
            delta = np.max(trans_prob * delta[:, :, None],
                           axis=1) * out_prob[:, :, output[i, j]]
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
    sns.heatmap(cm, annot=True, cbar=False, cmap="binary")
    plt.title(algorithm + f"\n(Accuracy : {acc}%)")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")


def main():
    parser = argparse.ArgumentParser()


if __name__ == "__main__":
    main()