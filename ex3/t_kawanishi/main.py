"""Regression analysis."""
import argparse
import csv
import re

import matplotlib.pyplot as plt
import numpy as np


# least squares method 2-dimension
def lsm_2(dataset: np.ndarray, deg: int, reg=False, stren=0.005) -> np.ndarray:
    """To adapt least squares method to 2-dimension dataset and return the decided degree function.

    Args:
        dataset (np.ndarray): As the name implies
        deg (int): the degree of the function
        reg (bool): decide whether to regularization or not to
        stren (float): the strength of the regularization

    Returns:
        np.ndarray: the coefficient of the function
    """
    # generate matrix for compute
    # to adapt (X^(T)X)^(-1)X^(T)Y
    X = np.array(
        [
            [np.power(dataset[0][j], i) for i in range(deg + 1)]
            for j in range(len(dataset[0]))
        ]
    )
    XT = X.T
    if reg:
        if stren < 0:
            raise ValueError("stren should greater than or equal to 0.")
        mat_I = np.eye(deg + 1)
        A = np.linalg.inv(XT @ X + stren * mat_I)  # (X^(T)X - λI)^(-1)
    else:
        A = np.linalg.inv(XT @ X)  # (X^(T)X)^(-1)

    return A @ XT @ dataset[1]


# least squares method 3-dimension
def lsm_3(dataset: np.ndarray, deg: int, reg=False, stren=0.005) -> np.ndarray:
    """To adapt least squares method to 3-dimension dataset and return the decided degree function.

    Args:
        dataset (np.ndarray): As the name implies
        deg (int): the degree of the function
        reg (bool): decide whether to regularization or not to
        stren (float): the strength of the regularization

    Returns:
        np.ndarray: the coefficient of the function
    """
    # create 3-dimension's least squares matrix
    powers = np.arange(deg + 1)
    X = np.column_stack(
        [
            np.power(dataset[0, :, np.newaxis], powers),
            np.power(dataset[1, :, np.newaxis], powers[1:]),
        ]
    )
    XT = X.T
    if reg:
        if stren < 0:
            raise ValueError("stren should greater than or equal to 0.")
        mat_I = np.eye(2 * deg + 1)
        A = np.linalg.inv(XT @ X + stren * mat_I)  # (X^(T)X - λI)^(-1)
    else:
        A = np.linalg.inv(XT @ X)  # (X^(T)X)^(-1)
    return A @ XT @ dataset[2]


# generate point to plot
def genpoint(
    dataset: np.ndarray, function: np.ndarray, quant=10000
) -> tuple[np.ndarray, np.ndarray]:
    """To generate function's data to plot.

    Args:
        dataset (np.ndarray): As the name implies
        function (np.ndarray): The function coefficient
        quant (int): The quantity of the data list. Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray]: function data list for plot
                                        first is x-aixs and next is f(x)-axis
    """
    # decide x1(x)-axis plot range
    x1_max = np.max(dataset[0])
    x1_min = np.min(dataset[0])

    # for 3-dim
    if len(dataset) == 3:
        # compute degree
        deg = int((len(function) - 1) / 2)
        # decide x2-axis plot range
        x2_max = np.max(dataset[1])
        x2_min = np.min(dataset[1])

        # generate point x1 and x2
        x1_group = np.linspace(x1_min, x1_max, int(np.sqrt(quant)))
        x2_group = np.linspace(x2_min, x2_max, int(np.sqrt(quant)))

        # generate mesh x1-x2
        x_group = np.stack(
            np.meshgrid(x1_group, x2_group)
        )  # x_group is a matrix (dim,quant,quant)

        # create outcome based on each value(cuz x and y are independent)
        x1_sum = np.sum(
            np.power(x1_group[:, np.newaxis], np.arange(deg + 1)) * function[: deg + 1],
            axis=1,
        )
        x2_sum = np.sum(
            np.power(x2_group[:, np.newaxis], np.arange(1, deg + 1))
            * function[deg + 1 : len(function)],
            axis=1,
        )

        # generate point f(x)
        f_group = x1_sum + x2_sum[:, np.newaxis]

    # for 2-dim
    else:
        # generate point x
        x_group = np.linspace(x1_min, x1_max, quant)

        # generate point f(x)
        f_group = np.sum(
            np.power(x_group[:, np.newaxis], np.arange(len(function))) * function,
            axis=1,
        )

    return x_group, f_group


if __name__ == "__main__":
    # get parser
    parser = argparse.ArgumentParser(
        description="""This program is to adapt least
                    squares method to the dataset for
                    regression analysis"""
    )
    parser.add_argument("path", help="The path of dataset")
    parser.add_argument(
        "-d",
        "--deg",
        help="The degree of the graph wants to generate",
        default=1,
        type=int,
    )
    parser.add_argument("-r", "--reg", help="adapt regularization", action="store_true")
    parser.add_argument(
        "-s",
        "--strength",
        help="the strength of the regularization",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-n", "--num", help="the amount of the plot point", default=10000, type=int
    )

    # read out parser
    args = parser.parse_args()

    # read scv file and change type
    data_array = []
    with open(args.path, "r") as f:
        reader = csv.reader(f)
        reader_data = next(reader)
        for line in reader:
            sub_group = []
            for value in line:
                sub_group.append(float(value))
            data_array.append(sub_group)

    # group by dimension
    data_array = np.array(data_array)
    data = data_array.T

    # plot data
    if len(data) == 2:
        # adapt lsm function
        func = lsm_2(data, args.deg, reg=args.reg, stren=args.strength)

        # create function label
        char = ""
        char += "{:.03f}".format(func[0])
        for i in range(1, args.deg + 1):
            if func[i] > 0:
                char += "+"
            char += str("{:.03f}".format(func[i])) + "$x^{" + str(i) + "}$"

        # generate point
        x_group, f_group = genpoint(data, func)

        # create graph and plot
        ax = plt.subplot()
        ax.plot(data[0], data[1], ".", label="dataset")
        ax.plot(x_group, f_group, label=char)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc=0)

    elif len(data) == 3:
        # adapt lsm function
        func = lsm_3(data, args.deg, reg=args.reg, stren=args.strength)

        # create function label
        char = ""
        char += "{:.03f}".format(func[0])
        for i in range(1, args.deg + 1):
            if func[i] > 0:
                char += "+"
            if i == 1:
                char += str("{:.03f}".format(func[i])) + "x"
            else:
                char += str("{:.03f}".format(func[i])) + "$x^{" + str(i) + "}$"
        for i in range(args.deg + 1, len(func)):
            if func[i] > 0:
                char += "+"
            if i - args.deg == 1:
                char += str("{:.03f}".format(func[i])) + "y"
            else:
                char += (
                    str("{:.03f}".format(func[i])) + "$y^{" + str(i - args.deg) + "}$"
                )

        # generate point
        x_group, f_group = genpoint(data, func, quant=args.num)

        # create graph and plot
        ax = plt.subplot(projection="3d")
        ax.plot(data[0], data[1], data[2], ".", c="b", label="dataset")
        ax.scatter(
            x_group[0], x_group[1], f_group, label=char, s=0.5, alpha=0.25, c="r"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

    else:
        raise ValueError(
            "data dimension should in 2 or 3 but " + str(len(data_array[0]))
        )

    # create image file
    img_num = re.sub(r"\D", "", args.path)
    ax.set_title("Dataset Num." + str(img_num))
    plt.savefig("graph" + img_num)
    plt.show()
