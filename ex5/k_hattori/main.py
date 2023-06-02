"""Perform vector quantization  and calculate MFCC for sound."""
import argparse
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct

import myfunc as mf


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


def clustering_2d(data, k, centroids):
    """
    Clustering data using centroids for 2D.

    Args:
        data (ndarray): Data array to be classified.
        k (int): Number of types to classify.
        centroids (ndarray): Centroids for vector quantization.

    Returns:
        ndarray: Classification result labels.
    """
    # Label for classification
    label = np.zeros(data.shape[0])
    # Quantize to nearest centroid in distance
    for i in range(data.shape[0]):
        distance = centroids - data[i]
        distance = distance[:, 0] ** 2 + distance[:, 1] ** 2
        label[i] = np.argmin(distance)
    return label


def k_means_2d(data, k):
    """
    Perform k_means method for 2D.

    Args:
        data (ndarray): Data array to be classified.
        k (int): Number of types to classify.

    Returns:
        ndarray: Classification result labels.
    """
    # Initialization of centroids
    rand_index = random.sample(range(data.shape[0]), k)
    centroids = data[rand_index]
    # Compute centroids until error converges.
    min_e = 0.03
    count = 0
    error = 1000
    while error > min_e and count < 1000:
        label = clustering_2d(data, k, centroids)
        previous_cent = centroids.copy()
        for i in range(k):
            centroids[i, 0] = np.mean(data[label == i, 0])
            centroids[i, 1] = np.mean(data[label == i, 1])
        difference = np.linalg.norm(np.abs(centroids - previous_cent), axis=1)
        error = difference[np.argmax(difference)]
        count += 1
    # Reclassify and plot vectors
    label = clustering_2d(data, k, centroids)
    for i in range(k):
        plt.scatter(data[label == i, 0], data[label == i, 1])
    plt.show()

    return label


def clustering_3d(data, k, centroids):
    """
    Clustering data using centroids for 3D.

    Args:
        data (ndarray): Data array to be classified.
        k (int): Number of types to classify.
        centroids (ndarray): Centroids for vector quantization.

    Returns:
        ndarray: Classification result labels.
    """
    # Label for classification
    label = np.zeros(data.shape[0])
    # Quantize to nearest centroid in distance
    for i in range(data.shape[0]):
        distance = centroids - data[i]
        distance = distance[:, 0]**2 + distance[:, 1]**2 + distance[:, 2]**2
        label[i] = np.argmin(distance)
    return label


def k_means_3d(data, k):
    """
    Perform k_means method for 3D.

    Args:
        data (ndarray): Data array to be classified.
        k (int): Number of types to classify.

    Returns:
        ndarray: Classification result labels.
    """
    # Initialization of centroids
    rand_index = random.sample(range(data.shape[0]), k)
    centroids = data[rand_index]
    # Compute centroids until error converges.
    min_e = 0.03
    count = 0
    error = 1000
    while error > min_e and count < 1000:
        label = clustering_3d(data, k, centroids)
        previous_cent = centroids.copy()
        for i in range(k):
            centroids[i, 0] = np.mean(data[label == i, 0])
            centroids[i, 1] = np.mean(data[label == i, 1])
            centroids[i, 2] = np.mean(data[label == i, 2])
        difference = np.linalg.norm(np.abs(centroids - previous_cent), axis=1)
        error = difference[np.argmax(difference)]
        count += 1
    # Reclassify and plot vectors
    label = clustering_3d(data, k, centroids)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(k):
        ax.scatter(data[label == i, 0], data[label == i, 1],
                   data[label == i, 2])
    plt.show()

    return label


def hz2mel(f):
    """
    Convert from frequency to mel scale.

    Args:
        f (float): Frequency to be converted to mel scale.

    Returns:
        float: Frequency converted to mel scale.
    """
    return 2595 * np.log(f / 700.0 + 1.0)


def mel2hz(m):
    """
    Convert from mel scale to frequency.

    Args:
        m (float): mel scale to be converted to frequency.

    Returns:
        float: mel scale converted to frequency.
    """
    return 700 * (np.exp(m / 2595) - 1.0)


def melFilterBank(channel, size, samplerate):
    """
    Create a mel filter Bank.

    Args:
        channel (int): A number of channels in mel filter bank.
        size (int): Filter size of mel filter bank.
        samplerate (float): sampling rate of data to be filtered.

    Returns:
        ndarray: Mel filter bank.
    """
    # Nyquist rate (frequency, mel and index)
    fmax = samplerate / 2
    melmax = hz2mel(fmax)
    nmax = size // 2
    # Frequency resolution
    delta_f = samplerate / size
    # mel width per channel
    delta_mel = melmax / (channel + 1)
    # Center frequency of filter
    centers_mel = np.arange(1, channel + 1) * delta_mel
    centers_freq = mel2hz(centers_mel)
    centers_index = np.round(centers_freq / delta_f)
    # Index of the start and stop position of each filter
    index_start = np.hstack(([0], centers_index[0: channel - 1]))
    index_stop = np.hstack((centers_index[0:channel], [nmax]))
    # create melfilterbank
    filterbank = np.zeros([channel, nmax])
    for i in range(0, channel):
        increment = 1.0 / (centers_index[i] - index_start[i])
        for j in range(int(index_start[i]), int(centers_index[i])):
            filterbank[i, j] = (j - index_start[i]) * increment
        decrement = 1.0 / (index_stop[i] - centers_index[i])
        for j in range(int(centers_index[i]), int(index_stop[i])):
            filterbank[i, j] = 1.0 - ((j - centers_index[i]) * decrement)

    return filterbank


def MFCC(data, length, samplerate):
    """
    Calculate MFCC of data.

    Args:
        data (ndarray): Data to be MFCC.
        length (int): Data segment width.
        samplerate (float): Sampling rate of data.

    Returns:
        ndarray: MFCC of calculation results.
    """
    order = 12
    N = len(data) // length
    hanning = np.hanning(length)
    mfcc = np.zeros([N, order])
    pos = 0
    for i in range(N):
        segment = data[pos: pos + length]
        segment = segment * hanning
        fft_seg = np.log10(np.abs(np.fft.fft(segment)))
        channel = 20
        filterbank = melFilterBank(channel, len(fft_seg), samplerate)
        # Convert to mel spectrum
        m_seg = np.dot(fft_seg[: len(fft_seg) // 2], filterbank.T)
        ceps = dct(m_seg, type=2, norm="ortho", axis=-1)
        mfcc[i] = ceps[:order]
        pos += length

    return mfcc


def delta_MFCC(mfcc):
    """Calculate variation of MFCC.

    Args:
        mfcc (ndarray): MFCC to be calculated.

    Returns:
        ndarray: Variation of MFCC.
    """
    d_mfcc = np.zeros(mfcc.shape)
    mfcc = np.append(mfcc, np.zeros([1, mfcc.shape[1]]), axis=0)
    for i in range(d_mfcc.shape[0]):
        d_mfcc[i] = mfcc[i + 1] - mfcc[i]

    return d_mfcc


def MFCC_plot(data, mfcc, d_mfcc, dd_mfcc, TOTAL_TIME, samplerate):
    """
    Plot MFCC on graphs.

    Args:
        data (ndarray): Spectrogram.
        mfcc (ndarray): MFCC.
        d_mfcc (ndarray): Delta-MFCC.
        dd_mfcc (ndarray): Delta-Delta-MFCC.
        TOTAL_TIME (float): Total time of data.
        samplerate (float): Sampling rate of data.

    Returns:
        None
    """
    freq = np.linspace(0, samplerate / 2, data.shape[1] // 2)
    amp = np.abs(data[:, data.shape[1] // 2 - 1:: -1])
    amp = 20 * np.log10(amp)

    plt.rcParams["image.cmap"] = "nipy_spectral"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10

    plt.figure(figsize=[8, 8])

    plt.subplot(411)
    plt.title("Spectrogram")
    plt.imshow(amp.T[freq <= 8000, :],
               extent=[0, TOTAL_TIME, 0, 8000], aspect="auto")
    plt.colorbar()
    plt.xlim(0, TOTAL_TIME)
    plt.ylim(0, 8000)
    plt.ylabel("Frequency [Hz]")

    plt.subplot(412)
    plt.title("MFCC sequence")
    plt.imshow(mfcc.T[::-1], extent=[0, TOTAL_TIME, 0, 12], aspect="auto")
    plt.colorbar()
    plt.xlim(0, TOTAL_TIME)
    plt.ylim(0, 12)
    plt.ylabel("MFCC")

    plt.subplot(413)
    plt.title("ΔMFCC sequence")
    plt.imshow(d_mfcc.T[::-1], extent=[0, TOTAL_TIME, 0, 12], aspect="auto")
    plt.colorbar()
    plt.xlim(0, TOTAL_TIME)
    plt.ylim(0, 12)
    plt.ylabel("ΔMFCC")

    plt.subplot(414)
    plt.title("ΔΔMFCC sequence")
    plt.imshow(dd_mfcc.T[::-1], extent=[0, TOTAL_TIME, 0, 12], aspect="auto")
    plt.colorbar()
    plt.xlim(0, TOTAL_TIME)
    plt.ylim(0, 12)
    plt.xlabel("Time [s]")
    plt.ylabel("ΔΔMFCC")

    plt.tight_layout()
    plt.show()


def main():
    """
    Read files and perform k_means method or calculate MFCC.

    Returns:
        None
    """
    # make parser
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="Demonstration of argparser",
        description="description",
        epilog="end",
        add_help=True,
    )
    # add arguments
    parser.add_argument("-f", dest="filename", help="Filename", required=True)
    parser.add_argument(
        "-k",
        dest="k",
        type=int,
        help="number for clustering",
        required=False,
        default=4,
    )
    # parse arguments
    args = parser.parse_args()

    path = args.filename
    # if args of -f is csv files.
    if path.find(".csv") != -1:
        data = load_csv(path)
        k = args.k
        dimension = np.shape(data)[1]

        if dimension == 2:
            k_means_2d(data, k)
        elif dimension == 3:
            k_means_3d(data, k)
    # if args of -f is wav files.
    elif path.find(".wav") != -1:
        data, samplerate = mf.wavload(path)
        TOTAL_TIME = len(data) / samplerate

        spec = mf.STFT(data, 1024)
        mfcc = MFCC(data, 2048, samplerate)
        d_mfcc = delta_MFCC(mfcc)
        dd_mfcc = delta_MFCC(d_mfcc)
        MFCC_plot(spec, mfcc, d_mfcc, dd_mfcc, TOTAL_TIME, samplerate)


if __name__ == "__main__":
    main()
    exit(1)
