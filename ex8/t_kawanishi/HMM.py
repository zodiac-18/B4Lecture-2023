"""Function of HMM."""
import pickle
import re
import time

import numpy as np


class HMM:
    def __init__(self, path):
        self.data = self.load_pickle(path)
        self.fname = self.get_fname(path)
        self.answer = np.array(self.data["answer_models"])
        self.output = np.array(self.data["output"])
        self.PI = np.array(self.data["models"]["PI"])
        self.A = np.array(self.data["models"]["A"])
        self.B = np.array(self.data["models"]["B"])
        if np.all(np.tril(self.A, k=-1) == 0):
            self.type = "left-to-right HMM"
        else:
            self.type = "Ergodic HMM"

    def get_fname(self, path: str) -> str:
        """Get file name.

        Args:
            path (str): path to the input file

        Returns:
            str: file name
        """
        f_name = re.sub(r".+\\", "", path)
        f_name = re.sub(r"\..+", "", f_name)
        return f_name

    def get_info(self):
        """get pickle information."""
        print()
        print((self.fname + "   " + self.type).center(40, "="))
        print("answer models of each sequence: "
              + str(self.answer.shape[0]).rjust(4))
        print("number of sequences: " + str(self.output.shape[0]).rjust(15))
        print("length of each sequence: "
              + str(self.output.shape[1]).rjust(11))
        print("Number of models: " + str(self.PI.shape[0]).rjust(18))
        print("Number of states: " + str(self.PI.shape[1]).rjust(18))
        print("Number of output states: " + str(self.B.shape[2]).rjust(11))
        print()

    def load_pickle(self, path: str) -> dict:
        """load pickle file

        Args:
            path (str): path to the pickle

        Returns:
            dict: pickle contents
        """
        data = pickle.load(open(path, "rb"))
        return data

    def Forward(self) -> tuple[np.ndarray, float, float]:
        """predict HMM by forward

        Returns:
            np.ndarray: predict
            float: 1 for-loop times
            float: 2 for-loop times
        """

        # ===============only used 1 for-loop==============
        start = time.perf_counter()
        output = self.output[:, :, np.newaxis]
        # initialize stage
        # shape is (models_n, status_n, sequence_n, 1)
        alpha = self.PI[:, :, np.newaxis] * self.B[:, :, output[:, 0]]

        # loop stage
        for i in range(1, self.output.shape[1]):
            alpha = (
                np.sum(alpha * self.A[:, :, np.newaxis], axis=1)
                .transpose(0, 2, 1)[:, :, :, np.newaxis]
                * self.B[:, :, output[:, i]]
            )

        pre = np.argmax(np.sum(alpha, axis=1), axis=0).reshape(-1)
        t1 = time.perf_counter() - start

        # ================used 2 for-loop====================
        start = time.perf_counter()
        pre1 = np.zeros_like(self.answer)
        for i in range(self.output.shape[0]):
            alpha = self.PI * self.B[:, :, output[i, 0]]
            for j in range(self.output.shape[1]):
                alpha = (
                    np.sum(alpha * self.A, axis=1)[:, :, np.newaxis]
                    * self.B[:, :, output[i, j]]
                )
            pre1[i] = np.argmax(np.sum(alpha, axis=1))
        t2 = time.perf_counter() - start

        return pre, t1 * 1000, t2 * 1000

    def Viterbi(self) -> tuple[np.ndarray, float, float]:
        """predict HMM by viterbi

        Returns:
            np.ndarray: predict
            float: 1 for-loop times
            float: 2 for-loop times
        """
        # ===============only used 1 for-loop==============
        start = time.perf_counter()
        output = self.output[:, :, np.newaxis]
        # initialize stage
        # shape is (models_n, status_n, sequence_n, 1)
        alpha = self.PI[:, :, np.newaxis] * self.B[:, :, output[:, 0]]

        # loop stage
        for i in range(1, self.output.shape[1]):
            alpha = (
                np.max(alpha * self.A[:, :, np.newaxis], axis=1)
                .transpose(0, 2, 1)[:, :, :, np.newaxis]
                * self.B[:, :, output[:, i]]
            )

        pre = np.argmax(np.max(alpha, axis=1), axis=0).reshape(-1)
        t1 = time.perf_counter() - start

        # ================used 2 for-loop====================
        start = time.perf_counter()
        pre1 = np.zeros_like(self.answer)
        for i in range(self.output.shape[0]):
            alpha = self.PI * self.B[:, :, output[i, 0]]
            for j in range(self.output.shape[1]):
                alpha = (
                    np.max(alpha * self.A, axis=1)[:, :, np.newaxis]
                    * self.B[:, :, output[i, j]]
                )
            pre1[i] = np.argmax(np.max(alpha, axis=1))
        t2 = time.perf_counter() - start

        return pre, t1 * 1000, t2 * 1000
