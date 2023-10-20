import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class HMM:
    def __init__(self, model, method="forward"):
        self.start_prob = np.array(model['PI'])
        self.transition_prob = np.array(model['A'])
        self.emission_prob = np.array(model['B'])
        self.method = method

    def predict(self, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if self.method == "forward":
            return self.forward_algorithm(y)
        elif self.method == "viterbi":
            return self.viterbi_algorithm(y)
        else:
            print(f'cannot support {self.method}')
            exit(0)

    def forward_algorithm(self, y):
        P, T = y.shape
        alpha = np.zeros((self.start_prob.shape[0], self.start_prob.shape[1], P))

        for i in range(P):
            # 初期確率を計算
            alpha[:, :, i] = self.start_prob[:, :, 0] * self.emission_prob[:, :, y[i, 0]]
            # 逐次計算
            for j in range(1, T):
                alpha[:, :, i] = np.sum(self.transition_prob.T * alpha[:, :, i].T, axis=1).T * self.emission_prob[:, :, y[i, j]]

        predicted = np.argmax(np.sum(alpha, axis=1), axis=0)
        return predicted

    def viterbi_algorithm(self, y):
        p, t = y.shape
        alpha = np.zeros((self.start_prob.shape[0], self.start_prob.shape[1], p))
        for i in range(p):
            alpha[:, :, i] = self.start_prob[:, :, 0] * self.emission_prob[:, :, y[i, 0]]
            for j in range(1, t):
                alpha[:, :, i] = np.max(self.transition_prob.T * alpha[:, :, i].T, axis=1).T * self.emission_prob[:, :, y[i, j]]

        predicted = np.argmax(np.max(alpha, axis=1), axis=0)
        return predicted


def confusion_matrix(answer, predicted):
    # create confusion matrix
    K = len(np.unique(answer))

    matrix = np.zeros((K, K), dtype=int)
    n_correct = 0
    for a, p in zip(answer, predicted):
        matrix[a, p] += 1
        if a == p:
            n_correct += 1
    acc = n_correct / len(answer)
    return matrix, acc


if __name__ == "__main__":
    fname = sys.argv[1]
    data = pickle.load(open(fname, "rb"))

    answer = data['answer_models']
    model = data['models']
    y = data['output']

    predicted_forward = HMM(model, method="forward").predict(y)
    predicted_viterbi = HMM(model, method="viterbi").predict(y)
    cm_forward, acc_forward = confusion_matrix(answer, predicted_forward)
    cm_viterbi, acc_viterbi = confusion_matrix(answer, predicted_viterbi)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    sns.heatmap(cm_forward, square=True, cbar=False, annot=True, cmap="Blues", ax=ax)
    ax.set_xlabel("predicted (forward)", fontsize=12)
    ax.set_ylabel("answer", fontsize=12)
    ax.set_title(f'forward algorithm\n Acc:{int(acc_forward*100)}%', fontsize=15)

    ax = fig.add_subplot(122)
    sns.heatmap(cm_viterbi, square=True, cbar=False, annot=True, cmap="Blues", ax=ax)
    ax.set_xlabel("predicted (viterbi)", fontsize=12)
    ax.set_ylabel("answer", fontsize=12)
    ax.set_title(f'viterbi algorithm\n Acc:{int(acc_viterbi*100)}%', fontsize=15)

    plt.tight_layout()
    plt.show()
