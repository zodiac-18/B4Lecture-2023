import numpy as np
import pandas as pd
import argparse
import random


def pick_random(data, k: int):
    """
    Pick random data.

    Args:
        data (ndarray): Data of csv file
        k (int): number of clustering
    Returns:
        ndarray: randomly picked data
    """
    index = random.sample(range(len(data)), k)
    value = data[index]
    return value


def k_means_2d(data, k: int):
    """
    Calculate k_means algorithm (2d).

    Args:
        data (ndarray): data of csv file
        k (int): number of clustering
    Returns:
        x (ndarray): Parameter of original data
        y (ndarray): Parameter of original data
        cluster_n (ndarray): cluster number
    """
    centroids = pick_random(data, k)
    x, y = data.T
    cluster_n = []

    for i in range(len(data)):
        dist = np.array([])
        for j in range(k):
            dist = np.append(dist, np.linalg.norm(data[i] - centroids[j]))
        cluster_n = np.append(cluster_n, np.argmin(dist))

    e_avg = 1.0
    # loop until new center equal pre center
    while e_avg != 0.0:
        sum = np.zeros(shape=(k, 2))
        e = 0
        for i in range(len(data)):
            for j in range(k):
                if cluster_n[i] == j:
                    sum[j] += data[i]

        n_points = np.array([])
        center = np.zeros(shape=(k, 2))
        for i in range(k):
            n_points = np.append(n_points, np.count_nonzero(cluster_n == i))
            # calculate average
            center[i][0] = sum[i][0] / n_points[i]
            center[i][1] = sum[i][1] / n_points[i]
        cluster_n = []

        for i in range(len(data)):
            dist = np.array([])
            for j in range(k):
                dist = np.append(dist, np.linalg.norm(data[i] - center[j]))
            cluster_n = np.append(cluster_n, np.argmin(dist))

        for j in range(k):
            e += np.linalg.norm(center[j] - centroids[j], ord=2)
        e_avg = e / k
        centroids = center

    return x, y, cluster_n


def k_means_3d(data, k: int):
    """
    Calculate k_means algorithm (2d).

    Args:
        data (ndarray): data of csv file
        k (int): number of clustering
    Returns:
        x (ndarray): Parameter of original data
        y (ndarray): Parameter of original data
        z (ndarray): Parameter of original data
        cluster_n (ndarray): cluster number
    """
    centroids = pick_random(data, k)
    x, y, z = data.T
    cluster_n = []

    for i in range(len(data)):
        dist = np.array([])
        for j in range(k):
            dist = np.append(dist, np.linalg.norm(data[i] - centroids[j]))
        cluster_n = np.append(cluster_n, np.argmin(dist))

    e_avg = 1.0
    # loop until new center equal pre center
    while e_avg != 0.0:
        sum = np.zeros(shape=(k, 3))
        e = 0
        for i in range(len(data)):
            for j in range(k):
                if cluster_n[i] == j:
                    sum[j] += data[i]

        n_points = np.array([])
        center = np.zeros(shape=(k, 3))
        for i in range(k):
            n_points = np.append(n_points, np.count_nonzero(cluster_n == i))
            # calcule average
            center[i][0] = sum[i][0] / n_points[i]
            center[i][1] = sum[i][1] / n_points[i]
            center[i][2] = sum[i][2] / n_points[i]

        cluster_n = []

        for i in range(len(data)):
            dist = np.array([])
            for j in range(k):
                dist = np.append(dist, np.linalg.norm(data[i] - center[j]))
            cluster_n = np.append(cluster_n, np.argmin(dist))

        for j in range(k):
            e += np.linalg.norm(center[j] - centroids[j], ord=2)
        e_avg = e / k
        centroids = center

    return x, y, z, cluster_n


def main():
    """Get calculated data from k_means algorithm."""
    parser = argparse.ArgumentParser(
        description='Program for getting calculated data from k_means algorithm.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-k", dest="k", type=int,
                        help='number for clustering', required=False, default=4)
    args = parser.parse_args()
    data = pd.read_csv(args.filename)
    data = np.array(data)
    k = args.k
    dimension = np.shape(data)[1]

    if dimension == 2:
        x, y = data.T
        k_means_2d(data, k)
    elif dimension == 3:

        k_means_3d(data, k)


if __name__ == "__main__":
    main()
