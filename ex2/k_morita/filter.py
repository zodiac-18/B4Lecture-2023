"""Filtering methods."""

import matplotlib.pyplot as plt
import numpy as np
import spectrogram as sp


def create_lpf(fc, sr, N, win_func=np.hamming):
    """Create LPF function.

    Args:
        fc (int): cut-off freqency
        sr (int): sample rate
        N (int): tap num
        win_func (function, optional): window function. Defaults to np.hamming.

    Returns:
        lpf (np.ndarray): low pass filter matrix
    """
    fc_norm = fc / (sr // 2)
    ideal_lpf = np.sinc(fc_norm * (np.arange(N) - (N - 1) / 2)) * fc_norm
    window = win_func(N)

    return ideal_lpf * window


def conv(input, filter):
    """Convolve function.

    Args:
        input (np.ndarray): input matrix
        filter (np.ndarray): filter matrix

    Returns:
        output (np.ndarray): convolved matrix
    """
    filter_len = len(filter)

    extra_arr = np.zeros(len(filter) - 1)
    input = np.concatenate([extra_arr, input, extra_arr])

    output_len = len(input) - len(extra_arr)
    output = np.zeros(output_len)

    filter = filter[::-1].T
    for i in range(output_len):
        left, right = i, i + filter_len
        output[i] = np.dot(input[left:right], filter)

    return output


if __name__ == "__main__":
    cutoff_freq = 2000
    samplerate = 16000
    nyquist_freq = samplerate // 2
    N = 101

    filter = create_lpf(cutoff_freq, samplerate, N)
    mag, phase = sp.magphase(np.fft.fft(filter))
    mag_db = sp.mag_to_db(mag)
    phase_deg = sp.rad_to_deg(phase)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = np.linspace(0, samplerate, len(mag_db))

    axes[0].plot(x, mag_db)
    axes[0].set_title("filter")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_xlim(0, nyquist_freq)

    axes[1].plot(x, phase_deg)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Phase [rad]")
    axes[1].set_xlim(0, nyquist_freq)
    axes[1].set_ylim(-200, 20)

    plt.tight_layout()
    plt.savefig("filter.png")

    # plt.show()
