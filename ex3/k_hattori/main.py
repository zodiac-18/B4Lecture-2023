"""Read files and do linear regression."""
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """
    Load csv files.

    Args:
        path (str): A file path.

    Returns:
        ndarray: Contents of csv file.
    """
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        buf = [row for row in reader]
    # Convert to ndarray in float64
    array = np.array(buf[1:])
    array = array.astype(np.float64)

    return array


def ridge_reg(x, y, n, k):
    """
    Perform ridge regression.

    Args:
        x (ndarray): An array of explanatory variables.
        y (ndarray): An array of target variables.
        n (int): Regression order.
        k (int): regularization factor.

    Returns:
        ndarray: Regression results.
    """
    if x.ndim == 1:
        x = x.reshape(-1, len(x)).T
    N = n * x.shape[1] + 1

    # Exponent part
    j = 0
    # Row number
    row = 0
    # Create a matrix of explanatory variables and identity matrix
    poly_x = np.zeros([x.shape[0], N])
    for i in range(N):
        # Reset index number and Handle next row(When x has 2 or more rows)
        if i and i % (n + 1) == 0:
            j = 1
            row += 1
        poly_x[:, i] = x[:, row] ** j
        j += 1

    # Identity matrix in order N
    matrix_I = np.eye(N)
    # Calculate a matrix of Regression coefficients beta
    tmp = np.dot(poly_x.T, poly_x)
    tmp = np.dot(np.linalg.inv(tmp + k * matrix_I), poly_x.T)
    beta = np.dot(tmp, y)

    # Calculate regression results
    y_predict = 0
    for i in range(N):
        y_predict += poly_x[:, i] * beta[i]
    if y_predict.ndim > 1:
        y_result = 0
        for i in range(y_predict.shape[1]):
            y_result += y_predict[:, i]
        y_result -= y_predict.shape[1] - 1

    return y_predict


def main():
    """
    Perform linear regressions and plot on graphs.

    Returns:
        None
    """
    # Load csv files
    data1 = load_csv(r"ex3\data1.csv")
    data2 = load_csv(r"ex3\data2.csv")
    data3 = load_csv(r"ex3\data3.csv")

    # Liniear regression
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    y1_predict = ridge_reg(x1, y1, 15, 0)
    y1_predict_regularized = ridge_reg(x1, y1, 15, 5)

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    y2_predict = ridge_reg(x2, y2, 6, 0)

    x3 = data3[:, 0]
    y3 = data3[:, 1]
    z3 = data3[:, 2]
    z3_predict = ridge_reg(data3[:, :2], z3, 2, 0)

    # Plot data
    plt.title("data1.csv")
    plt.scatter(x1, y1, color="silver", label="data1")
    plt.plot(
        x1[x1.argsort()], y1_predict[x1.argsort()],
        color="orange", label="regression"
    )
    plt.plot(
        x1[x1.argsort()],
        y1_predict_regularized[x1.argsort()],
        color="blue",
        label="ridge regression"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    plt.title("data2.csv")
    plt.scatter(x2, y2, color="silver", label="data2")
    plt.plot(
        x2[x2.argsort()], y2_predict[x2.argsort()],
        color="orange", label="regression"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("data3")
    ax.scatter(x3, y3, z3, color="silver", label="data3")
    ax.scatter(x3, y3, z3_predict, marker="+",
               color="orange", label="regression")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    plt.show()


if __name__ == "__main__":
    main()
    exit(1)
