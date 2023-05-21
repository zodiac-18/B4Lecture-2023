import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D


def lstsq(x, y, degree:int, demention:int):
    """least-squares method

    Args:
        x (ndarray): explanatory variables
        y (ndarray): objective variables
        degree (int): degree
        dimention (int):

    Returns:
        ndarray: regression coefficient
    """
    A_t = x ** degree
    for i in range(degree-1, -1, -1):
        A_t = np.vstack((A_t, x ** i))
    if demention > 2:
        print(1)
        A_t = np.delete(A_t, slice(-1, -demention+1, -1), 0)
    A = A_t.T
    v = np.array(y).T
    print(A)
    u = np.linalg.inv(A.T @ A) @ A.T @ v

    return u



def main():
    # read csv file
    data1 = np.genfromtxt("../data1.csv", delimiter=",", skip_header=1)
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    
    data2 = np.genfromtxt("../data2.csv", delimiter=",", skip_header=1)
    x2 = data2[:, 0]
    y2 = data2[:, 1]

    data3 = np.genfromtxt("../data3.csv", delimiter=",", skip_header=1)
    x3 = data3[:, 0]
    y3 = data3[:, 1]
    z3 = data3[:, 2]

    u1 = lstsq(x1, y1, 1, 2)
    u2 = lstsq(x2, y2, 3, 2)
    u3 = lstsq(np.vstack((x3, y3)), z3, 2, 3)


    # scatter plot
    fig, ax1 = plt.subplots(2, 1, tight_layout=True)
    ax1[0].scatter(x1, y1, label="Observed data")
    ax1[1].scatter(x2, y2)
    y1_label = "y=" + str(u1[0]) + "x" + str(u1[1])
    y2_label = "y=" + str(u2[0]) + "x^2" + str(u2[1]) + "x" + str(u2[2])
    ax1[0].plot(x1, x1 * u1[0] + u1[1], label=y1_label)
    ax1[1].plot(x2, x2 ** 3 * u2[0] + x2 ** 2 * u2[1] + x2 * u2[2] + u2[3], label=y2_label)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1, projection="3d")
    ax2.scatter(x3, y3, z3)

    plt.show()



if __name__ == "__main__":
    main()