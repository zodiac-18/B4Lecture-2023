"Main."
import argparse

import librosa
import librosa.display as lidi
import matplotlib.pyplot as plt
import numpy as np

import k_means as km
import mfcc


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform k-means and mfcc")
    parser.add_argument(
        "--csv_file",
        type=str,
        help="data csv file",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=2,
        help="number of cluster",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default="512",
        help="number of FFT points",
    )
    parser.add_argument(
        "--ceps",
        type=int,
        default="12",
        help="number of cepstrum",
    )
    parser.add_argument(
        "--wav_file",
        type=str,
        help="data wav file",
    )

    return parser.parse_args()


def open_csv(file_path):
    """Read csv file.
    Args:
        file_path (str): Csv file to read
    Returns:
        ndarray: Data read
    """
    data_set = np.loadtxt(fname=file_path, delimiter=",", skiprows=1)

    return data_set


def scat_plot2d(data, k):
    """Plot two-dimensional data.
    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        beta_r (ndarray): Regularized regression coefficient
        beta (ndarray): regression coefficient
    """
    labels, _ = km.k_means2d(data, k, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    ax.set_title("K-Means 2d")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.savefig('result/Kmeans.png')
    plt.show()
    plt.close()


def scat_plot3d(data, k):
    """Plot three-dimensional data.
    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        z (ndarray): data of z axis
        beta_r (ndarray): Regularized regression coefficient
        beta (ndarray): regression coefficient
        N1 (int) : dimension of x
        N2 (int) : dimension of y
    """
    labels, _ = km.k_means3d(data, k, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("K-Means 3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    plt.savefig('result/Kmeans.png')
    plt.show()


def mel_plot(data, fs, mfcc, d, dd):
    """Plot melspectrum.
    Args:
        data (nadarray): target data
        fs (int): sampling frequency
        mfcc (ndarray): mfcc data
        d (ndarray): delta data
        dd (ndarray): delta delta data
    """
    # plot original spectrogram
    plt.subplot(111)
    plt.specgram(data, fs)
    plt.colorbar()
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.savefig("result/original_spec.png")
    plt.show()
    plt.clf()
    plt.close()

    # plot mfcc, delta, deltadelta
    plt.subplot(311)
    lidi.specshow(mfcc.T)
    plt.colorbar()
    plt.ylabel("MFCC")

    plt.subplot(312)
    lidi.specshow(d.T)
    plt.colorbar()
    plt.ylabel("Delta")

    plt.subplot(313)
    lidi.specshow(dd.T)
    plt.colorbar()
    plt.ylabel("Delta delta")
    plt.xlabel("time")
    plt.savefig("result/mfcc.png")

    plt.show()


def main():
    """Regression analysis using the least squares method."""
    args = parse_args()

    file_path = args.csv_file
    k = args.cluster

    data = open_csv(file_path)

    if data.shape[1] == 2:
        scat_plot2d(data, k)

    elif data.shape[1] == 3:
        scat_plot3d(data, k)

    data_wav, fs = librosa.load(args.wav_file, mono=True)
    nfft = args.nfft
    ceps = args.ceps

    mfcc_data = mfcc.mfcc(data_wav, fs, 40, nfft, ceps)
    d = mfcc.delta(mfcc_data)
    dd = mfcc.delta(d)
    mel_plot(data_wav, fs, mfcc_data, d, dd)


if __name__ == "__main__":
    main()