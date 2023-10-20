"""Perform MFCC analysis"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack import dct


def load_sound_file(filename):
    """
    Load sound file.

    Args:
        filename (str): file name

    Returns:
        data (ndarray): sound data
        sr (int): sample rate
    """
    data, sr = librosa.load(filename, sr=None)
    return data, sr


def delta(data):
    """
    Calculate delta of data.

    Args:
        data (ndarray): Data to calculate
    """
    delta_result = []
    data = data.T
    for data_per in data:
        delta_result.append([])
        delta_result[-1].append(0)
        for i in range(1, len(data_per) - 1):
            # Use the difference from the previous result
            delta_result[-1].append(data_per[i] - data_per[i - 1])
        delta_result[-1].append(0)
    return np.array(delta_result).T


def convert_heltz_to_mel(f):
    """
    convert heltz to mel

    Args:
        f (int): Nyquist frequency (Hz)
    """
    return 2595 * np.log(f / 700.0 + 1.0)


def convert_mel_to_heltz(m):
    """
    convert mel to heltz

    Args:
        m (ndarray): mel
    """
    return 700 * (np.exp(m / 2595) - 1.0)


def mel_filter_bank(fs, N, num_channels):
    """
    Make mel_filter_bank

    Args:
        fs (int): sample rate
        N (int): _window length
        num_channels (_type_): number of channels

    Returns:
        filterbank (list): result of mfcc
    """
    # Nyquist frequency (Hz)
    fmax = fs / 2
    #  Nyquist frequency（mel）
    melmax = convert_heltz_to_mel(fmax)
    # Maximum number of frequency indexes
    nmax = N // 2
    # Frequency resolution (Hz width per frequency index 1)
    df = fs / N
    # Find the center frequency of each filter in the mel scale
    dmel = melmax / (num_channels + 1)
    melcenters = np.arange(1, num_channels + 1) * dmel
    print(type(melcenters))
    # Convert the center frequency of each filter to Hz
    fcenters = convert_mel_to_heltz(melcenters)
    # convert center frequency of each filter to frequency index
    indexcenter = np.round(fcenters / df)
    # Index the start position of each filter
    indexstart = np.hstack(([0], indexcenter[0: num_channels - 1]))
    # index of the end position of each filter
    indexstop = np.hstack((indexcenter[1:num_channels], [nmax]))
    filterbank = np.zeros((num_channels, nmax))
    for c in range(0, num_channels):
        # Find a point from the slope of the line to the left of the triangular filter
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # Find the point from the slope of the line to the right of the triangular filter
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank


def calc_mfcc(data, sr, win_length=1024, hop_length=512, mfcc_dim=12):
    """
    Calculate mfcc

    Args:
        data (list): original data of wav.file
        sr (int): sample rate
        win_length (int, optional): length of window function. Defaults to 1024.
        hop_length (int, optional): length of hop. Defaults to 512.
        mfcc_dim (int, optional): . Defaults to 12.

    Returns:
        mfcc (list): result of mfcc
    """
    data_length = data.shape[0]
    window = np.hamming(win_length)

    mfcc = []
    for i in range(int((data_length - hop_length) / hop_length)):
        # Cut data
        tmp = data[i * hop_length: i * hop_length + win_length]
        # Apply window function
        tmp = tmp * window
        # Apply FFT
        tmp = np.fft.rfft(tmp)
        # Get power spectrum
        tmp = np.abs(tmp)
        tmp = tmp[: win_length // 2]

        # Filter banks
        channels_n = 20
        filterbank = mel_filter_bank(sr, win_length, channels_n)
        # Apply filterbank
        tmp = np.dot(filterbank, tmp)
        # log
        tmp = 20 * np.log10(tmp)
        # Convert cosine transform
        tmp = dct(tmp, norm="ortho")
        # Apply the lifter
        tmp = tmp[1: mfcc_dim + 1]

        mfcc.append(tmp)

    mfcc = np.transpose(mfcc)
    return mfcc


def mfcc_plot(filename: str):
    """
    Plot mfcc

    Args:
        filename (str): data name of soundfile
    """
    data, fs = librosa.load(filename)

    win_length = 512
    hop_length = 256

    spectrogram = librosa.stft(data, win_length=win_length, hop_length=hop_length)
    spectrogram_db = 20 * np.log10(np.abs(spectrogram))

    fig = plt.figure(figsize=(12, 10))

    ax0 = fig.add_subplot(411)
    img = librosa.display.specshow(
        spectrogram_db, y_axis="log", sr=fs, cmap="rainbow", ax=ax0
    )
    ax0.set_title("Spectrogram")
    ax0.set_ylabel("frequency [Hz]")
    fig.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax0, format="%+2.f dB")

    # display mfcc
    mfcc_dim = 12
    ax1 = fig.add_subplot(412)
    mfcc = calc_mfcc(data, fs, win_length, hop_length, mfcc_dim)
    wav_time = data.shape[0] // fs
    extent = [0, wav_time, 0, mfcc_dim]
    img1 = ax1.imshow(np.flipud(mfcc), aspect="auto", extent=extent, cmap="rainbow")
    ax1.set_title("MFCC sequence")
    ax1.set_ylabel("MFCC")
    ax1.set_yticks(range(0, 13, 2))
    fig.colorbar(img1, aspect=10, pad=0.01, extend="both", ax=ax1, format="%+2.f dB")

    # display Δmfcc
    ax2 = fig.add_subplot(413)
    dmfcc = delta(mfcc)
    img2 = ax2.imshow(np.flipud(dmfcc), aspect="auto", extent=extent, cmap="rainbow")
    ax2.set(title="ΔMFCC sequence", ylabel="ΔMFCC", yticks=range(0, 13, 2))
    fig.colorbar(img2, aspect=10, pad=0.01, extend="both", ax=ax2, format="%+2.f dB")

    # display ΔΔmfcc
    ax3 = fig.add_subplot(414)
    ddmfcc = delta(dmfcc)
    img3 = ax3.imshow(np.flipud(ddmfcc), aspect="auto", extent=extent, cmap="rainbow")
    ax3.set(
        title="ΔΔMFCC sequence",
        xlabel="time[s]",
        ylabel="ΔΔMFCC",
        yticks=range(0, 13, 2),
    )
    fig.colorbar(img3, aspect=10, pad=0.01, extend="both", ax=ax3, format="%+2.f dB")

    fig.tight_layout()
    fig.savefig("mfcc.png")


def main():
    """
    Perform mfcc
    """
    parser = argparse.ArgumentParser(
        description="Perform MFCC analysis."
    )

    parser.add_argument("-f", "--filename", help="File name")

    args = parser.parse_args()

    filename = args.filename

    mfcc_plot(filename)


if __name__ == "__main__":
    main()
