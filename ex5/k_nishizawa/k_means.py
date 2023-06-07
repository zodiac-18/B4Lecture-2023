"K-Means algorithm."
import numpy as np
from numpy.random import PCG64, Generator


def random_cent2d(data, k):
    """Generate first centroids.
    Args:
        data (ndarray): target data
        k (int): number of divisions
    Returns:
        ndarray: array of centroids
    """
    dimension = data.shape[1]
    random = np.zeros((k, dimension))
    rg_pcg = PCG64()
    generator = Generator(rg_pcg)

    min_x = np.min(data[:, 0], axis=0)
    max_x = np.max(data[:, 0], axis=0)
    min_y = np.min(data[:, 1], axis=0)
    max_y = np.max(data[:, 1], axis=0)
    random_x = generator.uniform(min_x, max_x, k)
    random_y = generator.uniform(min_y, max_y, k)
    for i in range(0, k):
        random[i, 0] = random_x[i]
        random[i, 1] = random_y[i]

    return random


def random_cent3d(data, k):
    """Generate first centroids.
    Args:
        data (ndarray): target data
        k (int): number of divisions
    Returns:
        ndarray: array of centroids
    """
    dimension = data.shape[1]
    random = np.zeros((k, dimension))
    rg_pcg = PCG64()
    generator = Generator(rg_pcg)

    min_x = np.min(data[:, 0], axis=0)
    max_x = np.max(data[:, 0], axis=0)
    min_y = np.min(data[:, 1], axis=0)
    max_y = np.max(data[:, 1], axis=0)
    min_z = np.min(data[:, 2], axis=0)
    max_z = np.max(data[:, 2], axis=0)
    random_x = generator.uniform(min_x, max_x, k)
    random_y = generator.uniform(min_y, max_y, k)
    random_z = generator.uniform(min_z, max_z, k)
    for i in range(0, k):
        random[i, 0] = random_x[i]
        random[i, 1] = random_y[i]
        random[i, 2] = random_z[i]

    return random


def k_means2d(data, K, max_iters):
    """Calculate kmeans of 2d.
    Args:
        data (ndarray): target data
        K (int): number of divisions
        max_iters (int): number of max loop
    Returns:
        ndarray: labels and centroids
    """

    centroids = random_cent2d(data, K)

    for _ in range(max_iters):
        # 各データ点を最も近いクラスタ中心に割り当てる
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 新しいクラスタ中心を計算
        new_centroids = np.array(
            [data[labels == k].mean(axis=0) for k in range(K)]
        )

        # クラスタ中心の変化が小さい場合は終了
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def k_means3d(data, K, max_iters):
    """Calculate kmeans of 3d.
    Args:
        data (ndarray): target data
        K (int): number of divisions
        max_iters (int): number of max loop
    Returns:
        ndarray: labels and centroids
    """

    centroids = random_cent3d(data, K)

    for _ in range(max_iters):
        # 各データ点を最も近いクラスタ中心に割り当てる
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 新しいクラスタ中心を計算
        new_centroids = np.array(
            [data[labels == k].mean(axis=0) for k in range(K)]
        )

        # クラスタ中心の変化が小さい場合は終了
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids