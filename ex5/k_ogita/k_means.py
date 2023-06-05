#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classify the data using k-means method."""
import argparse
import collections
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, cluster_n, dim=2):
        self.cluster_n = cluster_n
        self.labels = None
        self.centroids = np.zeros((cluster_n, dim))

    def fit(self, data):
        N = data.shape[0]
        cluster_n = self.cluster_n
        # 1. Assign labels to data randomly
        cycle_labels = [(i % cluster_n) for i in range(N)]
        self.labels = np.array(random.sample(cycle_labels, N))
        labels_prev = np.zeros(N)
        while not (self.labels == labels_prev).all():
            # 2. Calculate centroids
            for i in range(cluster_n):
                indices = np.where(self.labels == i)
                x_i = data[indices, :]
                self.centroids[i] = np.mean(x_i, axis=1)
            # 3. Cluster reclassification
            labels_prev = self.labels
            # Calculate the distance to each centroid and label the nearest claster
            for i in range(N):
                self.labels[i] = np.argmin(
                    [
                        np.linalg.norm(data[i, :] - self.centroids[j, :], ord=2)
                        for j in range(cluster_n)
                    ]
                )

    def predict(self, data):
        N = data.shape[0]
        labels = np.zeros(N)
        for i in range(N):
            labels[i] = np.argmin(
                [
                    np.linalg.norm(data[i, :] - self.centroids[j, :], ord=2)
                    for j in range(self.cluster_n)
                ]
            )
        return labels


def main():
    """Classify the data using k-means method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
    parser.add_argument(
        "-c", "--cluster_n", help="The number of cluster", type=int, required=True
    )
    parser.add_argument(
        "-n",
        "--min_cluster_data_num",
        help="Minimum number of data in each cluster",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-save", "--save_fig", help="Whether to save figure or not", action="store_true"
    )
    args = parser.parse_args()

    # Get the path of data
    path = args.path
    # Get the filename of data from the path
    file_name = os.path.splitext(os.path.basename(path))[0]

    # Read data from the csv file
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(list(map(float, row)))
    data = np.array(data)
    # The number of cluster.
    cluster_n = args.cluster_n

    min_cluster_data_num = args.min_cluster_data_num

    # Get the dimention of the data.
    dim = data.shape[1]

    cmap_keyword = "jet"
    cmap = plt.get_cmap(cmap_keyword)

    model = KMeans(cluster_n, dim)
    count_labels = np.zeros(data.shape[0])
    while not (
        np.all(count_labels > min_cluster_data_num) and len(count_labels) == cluster_n
    ):
        model.fit(data)
        labels = model.labels
        count_labels = np.array(list(collections.Counter(labels).values()))
    scatter_label_list = [f"cluster_{k:0=2}" for k in range(cluster_n)]

    fig = plt.figure(figsize=(15, 10))

    # k-means clustering　for two-dimensional data
    if dim == 2:
        ax = fig.add_subplot(111)
        for i in range(cluster_n):
            indices = np.where(labels == i)[0]
            ax.scatter(
                data[indices, 0],
                data[indices, 1],
                color=cmap((i + 1) / cluster_n),
                label=scatter_label_list[i],
            )
    # k-means clustering　for three-dimensional data
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        for i in range(cluster_n):
            indices = np.where(labels == i)[0]
            ax.scatter3D(
                data[indices, 0],
                data[indices, 1],
                data[indices, 2],
                color=cmap((i + 1) / cluster_n),
                label=scatter_label_list[i],
            )
        ax.set_zlabel("$z$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"$k$-means clustering of {file_name} ($k={cluster_n}$)")
    ax.legend(loc="upper left")
    if args.save_fig:
        fig.savefig(f"{file_name}_{cluster_n}.png")
    plt.show()


if __name__ == "__main__":
    main()
