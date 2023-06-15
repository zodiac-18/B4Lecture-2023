import argparse
import numpy as np
import matplotlib.pyplot as plt


def Initialize(data, k):
    Dim = data.shape[1]
    Mu = np.random.randn(k, Dim)
    Sigma = np.array([np.identity(Dim) for i in range(k)])
    Pi = np.ones(k) / k

    return Mu, Sigma, Pi


def normal_dist(data, Mu, Sigma):
    Dim = data.shape[1]
    data_0 = data - Mu[:, None]
    tmp = data_0 @ np.linalg.inv(Sigma) @ data_0.transpose(0, 2, 1)
    tmp = tmp.diagonal(axis1=1, axis2=2)

    denom = np.sqrt((2 * np.pi) ** Dim) * np.sqrt(np.linalg.det(Sigma))
    numer = np.exp(-1 * tmp / 2)

    gauss = numer / denom[:, None]
    return gauss


def gmm(data, Mu, Sigma, Pi):
    weighted_gauss = normal_dist(data, Mu, Sigma) * Pi[:, None]
    mixed_gauss = np.sum(weighted_gauss, axis=0)

    return mixed_gauss, weighted_gauss


def EM_algorithm(data, Mu, Sigma, Pi, epsilon=1e-3):
    N = data.shape[0]
    Dim = data.shape[1]
    mixed_gauss, weighted_gauss = gmm(data, Mu, Sigma, Pi)

    log_likelifood = np.sum(np.log(mixed_gauss))
    Diff = 100
    while Diff > epsilon:
        gamma = weighted_gauss / mixed_gauss

        nk = np.sum(gamma, axis=1)
        Mu = np.sum(gamma[:, :, None] * data / nk[:, None, None], axis=1)
        data_0 = data - Mu[:, None]
        Sigma = (gamma[:, None] * data_0.transpose(0, 2, 1) @ data_0) / nk[:, None, None]
        Pi = nk / N

        mixed_gauss, weighted_gauss = gmm(data, Mu, Sigma, Pi)
        log_newlikelifood = np.sum(np.log(mixed_gauss))

        Diff = log_newlikelifood - log_likelifood
        log_likelifood = log_newlikelifood

    return Mu, Sigma, Pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()



if __name__ == "__main__":
    main()