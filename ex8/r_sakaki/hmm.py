import pickle
import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def forward(output, A, B, PI):
    number_of_outputs = output.shape[0]
    length_of_output = output.shape[1]
    predict_models = np.zeros(number_of_outputs)
    for i in range(number_of_outputs):
        # 初期化
        output_o1 = output[i, 0]
        alpha = PI[:, :, 0] * B[:, :, output_o1]
        # 再帰
        for j in range(1, length_of_output):
            output_oi = output[i, j]
            alpha = np.sum(alpha[:, :, np.newaxis] * A, axis=1)\
                * B[:, :, output_oi]
        # モデル毎の確率計算
        P = np.sum(alpha, axis=1)
        predict_models[i] = np.argmax(P)
    # 予想したモデル番号の配列を返す
    return predict_models


def viterbi(output, A, B, PI):
    number_of_outputs = output.shape[0]
    length_of_output = output.shape[1]
    predict_models = np.zeros(number_of_outputs)
    for i in range(number_of_outputs):
        # 初期化
        output_o1 = output[i, 0]
        alpha = PI[:, :, 0] * B[:, :, output_o1]
        # 再帰
        for j in range(1, length_of_output):
            output_oi = output[i, j]
            alpha = np.max(alpha[:, :, np.newaxis] * A, axis=1)\
                * B[:, :, output_oi]
        # モデル毎の確率計算
        P = np.max(alpha, axis=1)
        predict_models[i] = np.argmax(P)
    # 予想したモデル番号の配列を返す
    return predict_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to input data")
    args = parser.parse_args()

    path = args.path
    filename = os.path.splitext(os.path.basename(path))[0]

    # データの取得
    data = pickle.load(open(path, "rb"))
    output = np.array(data["output"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])
    PI = np.array(data["models"]["PI"])
    answer_models = np.array(data["answer_models"])

    # HMM実行
    predicted_models_forward = forward(output, A, B, PI)
    predicted_models_viterbi = viterbi(output, A, B, PI)

    # グラフ表示
    plt.rcParams["figure.figsize"] = (13, 6)
    fig = plt.figure()
    # forwardアルゴリズム
    plt.subplot(121)
    acc_forward = 100 * accuracy_score(answer_models, predicted_models_forward)
    cm = confusion_matrix(answer_models, predicted_models_forward)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.title(f"Forward algorithm\n(Acc. {acc_forward:.0f}%)")

    # viterbiアルゴリズム
    plt.subplot(122)
    acc_viterbi = 100 * accuracy_score(answer_models, predicted_models_viterbi)
    cm = confusion_matrix(answer_models, predicted_models_viterbi)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.title(f"Viterbi algorithm\n(Acc. {acc_viterbi:.0f}%)")

    fig.savefig(f"result_{filename}.png")


if __name__ == "__main__":
    main()
