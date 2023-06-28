"""Ex8."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn


def plot_mesh(data: np.ndarray, ax, ax_index: int, title: str):
    """Plot data as mesh.

    Args:
        data (np.ndarray): Data to data.
        ax (_type_): Ax.
        ax_index (int): The index of data.
        title (str): The title of the figure.
    """
    seaborn.heatmap(data, annot=True, cbar=False, cmap="Blues", ax=ax[ax_index])
    ax[ax_index].set_title(title)
    ax[ax_index].set_xlabel("Predicted")
    ax[ax_index].set_ylabel("Answer")


class HMM:
    def __init__(self, data: dict) -> None:
        self.answer_models = np.array(data["answer_models"])
        self.output = np.array(data["output"])
        self.PI = np.array(data["models"]["PI"])
        self.A = np.array(data["models"]["A"])
        self.B = np.array(data["models"]["B"])
        self.p = self.output.shape[0]  # 出力系列数(推測するモデル数)
        self.t = self.output.shape[1]  # 出力系列の長さ(時間)
        self.k = self.B.shape[0]  # モデル数
        self.l = self.B.shape[1]  # 状態数
        self.n = self.B.shape[2]  # 出力の種類数
        self.model_forward = None  # forwardアルゴリズムの結果
        self.model_viterbi = None  # forwardアルゴリズムの結果

    def forward(self):
        """Forward Algorithm."""
        likelihood = np.ones((self.p, self.k, self.l, 2))
        for p in range(self.p):
            # 初期状態(i=0)
            likelihood[p, :, :, 0] *= self.PI[:, :, 0] * self.B[:, :, self.output[p, 0]]
            # i=1 ~ i=tまで
            for i in range(1, self.t):
                likelihood[p, :, :, 1] = (
                    np.sum(likelihood[p, :, :, 0].T * self.A.T, axis=1).T
                    * self.B[:, :, self.output[p, i]]
                )
                likelihood[p, :, :, 0] = likelihood[p, :, :, 1]
        select_model = np.argmax(np.sum(likelihood[:, :, :, 1], axis=2), axis=1)
        self.model_forward = select_model

    def viterbi(self):
        """Viterbi Algorithm."""
        likelihood = np.ones((self.p, self.k, self.l, 2))
        for p in range(self.p):
            # 初期状態(i=0)
            likelihood[p, :, :, 0] *= self.PI[:, :, 0] * self.B[:, :, self.output[p, 0]]
            # i=1 ~ i=tまで
            for i in range(1, self.t):
                likelihood[p, :, :, 1] = (
                    np.max(likelihood[p, :, :, 0].T * self.A.T, axis=1).T
                    * self.B[:, :, self.output[p, i]]
                )
                likelihood[p, :, :, 0] = likelihood[p, :, :, 1]
        select_model = np.argmax(np.max(likelihood[:, :, :, 1], axis=2), axis=1)
        self.model_viterbi = select_model

    def fit(self, data_name: str):
        """Calculate HMM.

        Args:
            data_name (str): The data name.
        """
        self.forward()
        self.viterbi()
        cm_forward = confusion_matrix(self.answer_models, self.model_forward)
        ac_forward = accuracy_score(self.answer_models, self.model_forward)
        cm_viterbi = confusion_matrix(self.answer_models, self.model_viterbi)
        ac_viterbi = accuracy_score(self.answer_models, self.model_viterbi)
        # plot
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        plot_mesh(cm_forward, ax, 0, f"Forward {data_name} Acc : {ac_forward * 100}%")
        plot_mesh(cm_viterbi, ax, 1, f"Viterbi {data_name} Acc : {ac_viterbi * 100}%")
        plt.show()
        fig.savefig(f"{data_name}.png")


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        prog="main.py",  # プログラム名
        usage="B4 Lecture Ex6.",  # プログラムの利用方法
        description="Principal Component Analysis.",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("data_path", help="Select File.")
    args = parser.parse_args()

    data = pickle.load(open("../" + args.data_path + ".pickle", "rb"))
    hmm = HMM(data)
    hmm.fit(args.data_path)

    plt.clf()
    plt.close()
