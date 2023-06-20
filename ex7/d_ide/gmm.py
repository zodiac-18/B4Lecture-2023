import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def scatter_1d(data, clu, vec, cov, pi, savename):
    """1D scatter plots.

    Args:
        data (ndarray): target data
        clu (int): number of cluster
        vec (ndarray): vector
        cov (ndarray): covariance
        pi (ndarray): list of pi
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cluster = len(vec)

    cmap = plt.get_cmap("tab10")
    for i in range(cluster):
        cdata = data[clu == i]
        x = cdata
        y = np.zeros(x.shape[0])
        ax.plot(x, y, label=i, marker="o", color=cmap(i))
    x = vec
    y = np.zeros(cluster)
    ax.plot(x, y, label="centroids", marker="x", color=cmap(i + 1))

    pos = np.linspace(np.min(data) - 1, np.max(data) + 1, 100)
    y = np.zeros(100)
    for k in range(cluster):
        y += pi[k] * multivariate_normal.pdf(pos, vec[k], cov[k])
    ax.plot(pos, y, label="GMM")

    ax.set_xlabel("x")
    ax.set_ylabel("gaussian distribution")
    plt.title(savename)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/gmm.png")
    plt.show()


def scatter_2d(data, clu, vec, cov, pi, savename):
    """2D scatter plots.

    Args:
        data (ndarray): target data
        clu (int): number of cluster
        vec (ndarray): vector
        cov (ndarray): covariance
        pi (ndarray): list of pi
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cluster = len(vec)

    cmap = plt.get_cmap("tab10")
    for i in range(cluster):
        cdata = data[clu == i]
        x = cdata[:, 0]
        y = cdata[:, 1]
        ax.plot(x, y, label=i, linestyle="None", marker="o", color=cmap(i))
    x = vec[:, 0]
    y = vec[:, 1]
    ax.plot(
        x,
        y,
        label="centroids",
        linestyle="None",
        marker="x",
        color=cmap(i + 1)
    )

    posx = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 100)
    posy = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 100)
    posx, posy = np.meshgrid(posx, posy)
    pos = np.dstack([posx, posy])
    z = np.array([np.squeeze(gmm(i_pos, vec, cov, pi)[1], 0) for i_pos in pos])

    plt.contour(posx, posy, z)
    plt.colorbar()

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.title(savename)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/gmm.png")
    plt.show()


def logplot(log_list, savename):
    """Plot the log-likelihood function.

    Args:
        log_list (ndarray): calculated log
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(log_list)
    ax.set_xlabel("count", fontsize=18)
    ax.set_ylabel("log likelihood function", fontsize=18)
    plt.title(savename, fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.show()


def minimax(data, cluster):
    """Calculate minimax method.

    Args:
        data (ndarray): target data
        cluster (int): number of cluster

    Returns:
        ndarray: centroids list
    """
    num, dim = data.shape
    cidx = []
    cidx = np.append(cidx, random.randint(0, num - 1))
    dis = np.zeros((cluster, num))
    cen = np.zeros((cluster, dim))
    for k in range(cluster):
        cen[k] = data[int(cidx[k])]
        r = np.sum((data - data[int(cidx[k])]) ** 2, axis=1)
        dis[k] = r

        cidx = np.append(cidx, np.argmax(np.min(dis[: k + 1], axis=0)))
    return cen


def kmean(data, cluster, cen):
    """Calculate k-means.

    Args:
        data (ndarray): target data
        cluster (int): number of cluster
        cen (ndarray): centroids list

    Returns:
        ndarray: new centroids list
    """
    num, dim = data.shape
    dis = np.zeros((cluster, num))
    newcen = np.zeros((cluster, dim))
    while True:
        for k in range(0, cluster):
            r = np.sum((data - cen[k]) ** 2, axis=1)
            dis[k] = r

        clu = np.argmin(dis, axis=0)

        for i in range(0, cluster):
            newcen[i] = data[clu == i].mean(axis=0)

        if np.allclose(cen, newcen) is True:
            break
        cen = newcen
    return newcen, clu


def ini(data, cluster):
    """Calculate initial value.

    Args:
        data (ndarray): target data
        cluster (int): number of cluster

    Returns:
        ndarray: vector, covariance, pi
    """
    num, dim = data.shape
    cen = minimax(data, cluster)
    vec, clu = kmean(data, cluster, cen)
    pi = np.zeros(cluster)
    cov = np.zeros((cluster, dim, dim))

    for k in range(cluster):
        pi[k] = data[clu == k].shape[0]
        cov[k] = np.cov(data[clu == k].T)

    pi = pi / np.sum(pi)
    return vec, cov, pi


def gauss_all(data, vec, cov):
    """Calculate all gauss.

    Args:
        data (ndarray): target data
        vec (ndarray): vector
        cov (ndarray): covariance

    Returns:
        ndarray: _description_
    """
    num, dim = data.shape
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    Nk = np.zeros(num)

    for i in range(num):
        a = ((2 * np.pi) ** (0.5 * dim)) * (det**0.5)
        b = -0.5 * (data[i] - vec)[None, :] @ inv @ (data[i] - vec)
        Nk[i] = np.exp(b) / a

    return Nk


def gmm(data, vec, cov, pi):
    """mixed Gaussian distribution.

    Args:
        data (ndarray): target data
        vec (ndarray): vector
        cov (ndarray): covariance
        pi (ndarray): pi list

    Returns:
        ndarray: calculated gauss list
    """
    k = len(vec)
    N = np.array([pi[i] * gauss_all(data, vec[i], cov[i]) for i in range(k)])

    return N, np.sum(N, axis=0)[None, :]


def log_likelihood(data, vec, cov, pi):
    """Calculate Log-likelihood function.

    Args:
        data (ndarray): target data
        vec (ndarray): vector
        cov (ndarray): covariance
        pi (ndarray): pi list

    Returns:
        int: sum of log
    """
    num, dim = data.shape
    _, N_sum = gmm(data, vec, cov, pi)
    logs = np.array([np.log(N_sum[0][i]) for i in range(num)])
    return np.sum(logs)


def EM(data, vec, cov, pi, eps):
    """Calculate EM Algorithm.

    Args:
        data (ndarray): target data
        vec (ndarray): vector
        cov (ndarray): covariance
        pi (ndarray): pi list
        eps (flow): max number

    Returns:
        ndarray: result of EM Algorithm
    """
    cluster = vec.shape[0]
    num, dim = data.shape
    count = 0
    log_list = []

    while True:
        old_log = log_likelihood(data, vec, cov, pi)
        N, N_sum = gmm(data, vec, cov, pi)
        cov = np.zeros((cluster, dim, dim))
        gamma = N / N_sum
        vec = (gamma @ data) / np.sum(gamma, axis=1)[:, None]

        for k in range(cluster):
            for n in range(num):
                dis = data[n] - vec[k]
                cov[k] += gamma[k][n] * dis[:, None] @ dis[None, :]

            cov[k] = cov[k] / np.sum(gamma[k])

        pi = np.sum(gamma, axis=1) / num

        new_log = log_likelihood(data, vec, cov, pi)
        log_dif = old_log - new_log
        log_list = np.append(log_list, log_dif)

        if np.abs(log_dif) < eps:
            return count, gamma, vec, cov, pi, log_list
        else:
            count += 1
            old_log = new_log
