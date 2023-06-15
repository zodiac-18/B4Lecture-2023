'Main.'
import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform pca")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="data1.csv",
        help="data csv file",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="compress dimension",
    )

    return parser.parse_args()


def open_csv(file_path):
    """Read csv file.
    Args:
        file_path (str): Csv file to read
    Returns:
        ndarray: Data read
    """
    data_set = np.loadtxt(fname=file_path, delimiter=",")

    return data_set


def plot2d(data):
    """Plot 2 dimension data.
    Args:
        data (ndarray): target data
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], label="original data")
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()
    plt.close()


def plot1d(data):
    """Plot 3 dimension data
    Args:
        data (ndarray): target data
    """
    y = np.zeros_like(data)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data, y, label="original data", s=5)
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    #plt.savefig("result/pca_data2.png")
    plt.show()
    plt.close()


# 対数尤度関数のプロット
def logplot(log_list, save):
    """
    log_list:list
            log likelihood function
    save:str
         save name
    """
    num = str(save)[0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(log_list)
    ax.set_xlabel("count", fontsize=18)
    ax.set_ylabel("log likelihood function", fontsize=18)
    plt.title("data" + num, fontsize=18)
    plt.grid()
    plt.tight_layout()
    if type(save) == str:
        logsave = "log" + save
        plt.savefig(logsave)
    plt.show()


def main():
    """Calculate PCA."""
    args = parse_args()

    file_path = args.csv_file
    data = open_csv(file_path)
    d_dim = data.shape

    if file_path == "data1.csv":
        plot1d(data)
    elif file_path == "data2.csv" or file_path == "data3.csv":
        plot2d(data)


if __name__ == "__main__":
    main()
