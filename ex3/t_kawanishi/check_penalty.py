"""Generate graph to check relationship between penalty and graph similarity."""
import argparse
import csv
import re

import matplotlib.pyplot as plt
import numpy as np

import main as F


def penaltyData(
    dataset: np.ndarray, degree_group: np.ndarray, lambda_group: np.ndarray
) -> np.ndarray:
    """To generate penalty relationship group for plot.

    Args:
        dataset (np.ndarray): As the name implies
        degree_group (np.ndarray): The group of degree group
        lambda_group (np.ndarray): The group of penalty group

    Returns:
        np.ndarray: matrix that used for plot and should be
        (len(degree_group), len(lambda_group):Lambda applicable difference value)
    """
    # generate error group matrix
    error_group = np.zeros((len(degree_group), len(lambda_group)))

    # 2-dimension
    if len(dataset) == 2:
        for i in range(len(degree_group)):
            for j in range(len(lambda_group)):
                coef = F.lsm_2(
                    dataset, degree_group[i], reg=True, stren=lambda_group[j]
                )
                f_group = np.sum(
                    np.power(dataset[0][:, np.newaxis], np.arange(len(coef))) * coef,
                    axis=1,
                )
                error = dataset[1] - f_group
                error_group[i][j] += error.T @ error
    # 3-dimension
    else:
        for i in range(len(degree_group)):
            for j in range(len(lambda_group)):
                coef = F.lsm_3(
                    dataset, degree_group[i], reg=True, stren=lambda_group[j]
                )
                deg = int((len(coef) - 1) / 2)
                x1_sum = np.sum(
                    np.power(dataset[0][:, np.newaxis], np.arange(deg + 1))
                    * coef[: deg + 1],
                    axis=1,
                )
                x2_sum = np.sum(
                    np.power(dataset[1][:, np.newaxis], np.arange(1, deg + 1))
                    * coef[deg + 1 : len(coef)],
                    axis=1,
                )
                f_group = x1_sum + x2_sum
                error = dataset[2] - f_group
                error_group[i][j] += error.T @ error
    return error_group


if __name__ == "__main__":
    # get parser
    parser = argparse.ArgumentParser(
        description="This program is to check relationship between penalty and graph similarity"
    )
    parser.add_argument("path", help="The path of dataset")
    parser.add_argument(
        "-ds",
        "--deg_start",
        help="The degree of the graph wants to compare start range",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-de",
        "--deg_end",
        help="The degree of the graph wants to compare end range",
        default=5,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--strength",
        help="the strength of the regularization penalty range",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-n", "--num", help="the amount of the plot point", default=100, type=int
    )

    # read out parser
    args = parser.parse_args()

    # read scv file
    data_array = []
    with open(args.path, "r") as f:
        reader = csv.reader(f)
        reader_data = next(reader)
        for line in reader:
            data_array.append(line)

    # change type
    for i in range(len(data_array)):
        for j in range(len(data_array[i])):
            data_array[i][j] = float(data_array[i][j])
    data_array = np.array(data_array)
    data = data_array.T

    # create penalty group
    lambda_group = np.linspace(0, args.strength, args.num)

    # create degree group
    if args.deg_end < args.deg_start:
        raise ValueError(
            "deg_start should lesser than deg_end but "
            + str(args.deg_start)
            + " > "
            + str(args.deg_end)
        )
    degree_group = np.arange(args.deg_start, args.deg_end + 1)

    # compute relationship
    y_group = penaltyData(data, degree_group, lambda_group)

    # file name generate
    img_num = re.sub(r"\D", "", args.path)

    # plot data
    ax = plt.subplot()
    for i in range(len(degree_group)):
        ax.plot(lambda_group, y_group[i], label="degree:" + str(degree_group[i]))
    ax.set_xlabel("λ")
    ax.set_ylabel("error")
    ax.set_title("data" + str(img_num) + " correlation between λ and error")
    ax.legend()
    plt.savefig("error" + img_num)
    plt.show()
