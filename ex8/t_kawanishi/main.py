"""To adapt HMM."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import HMM


def main():
    """Main function."""
    # create parser
    parser = argparse.ArgumentParser(
        description="This is a program to model prediction by HMM"
    )
    parser.add_argument("path", help="path to the file")

    # get arguments
    args = parser.parse_args()

    # read out pickle
    """
    data
        answer_models (100,)
        output (100,100)
        models
            PI (5,3,1) or (5,5,1)
            A (5,3,3) or (5,5,5)
            B (5,3,5) or (5,5,5)
    """
    data = HMM.HMM(args.path)
    data.get_info()
    preF, tF1, tF2 = data.Forward()
    preV, tV1, tV2 = data.Viterbi()
    cmF = confusion_matrix(data.answer, preF)
    cmV = confusion_matrix(data.answer, preV)
    Acu_F = np.trace(cmF) / np.sum(cmF)
    Acu_V = np.trace(cmV) / np.sum(cmV)

    # plot confusion matrix
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(data.fname + " learning outcome\nData type: " + data.type, fontsize=20)
    A = fig.add_subplot(1, 2, 1)
    B = fig.add_subplot(1, 2, 2)
    sns.heatmap(cmF, ax=A, cmap="Purples", annot=True, square=True)
    sns.heatmap(cmV, ax=B, cmap="Purples", annot=True, square=True)
    A.set_title(
        "Forward "
        + data.fname
        + "\nAccuracy: "
        + "{:.3f}".format(Acu_F)
        + "\n1 for-loop time: "
        + "{:.3f}".format(tF1)
        + "ms     2 for-loop time: "
        + "{:.3f}".format(tF2)
        + "ms"
    )
    B.set_title(
        "Viterbi "
        + data.fname
        + "\nAccuracy: "
        + "{:.3f}".format(Acu_V)
        + "\n1 for-loop time: "
        + "{:.3f}".format(tV1)
        + "ms     2 for-loop time: "
        + "{:.3f}".format(tV2)
        + "ms"
    )
    A.set_xlabel("predicted models", fontsize=16)
    B.set_xlabel("predicted models", fontsize=16)
    A.set_ylabel("answer models", fontsize=16)
    B.set_ylabel("answer models", fontsize=16)
    plt.savefig(data.fname + "_confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
