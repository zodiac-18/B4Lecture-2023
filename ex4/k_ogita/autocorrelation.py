#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Estimate f0 frequency."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def calc_ac(x, N):
    """
    Calculate the autocorrelation of the frame.

    Args:
        x (ndarray): Input wave.
        N (int): The size of the frame.

    Returns:
        ndarray: Autocorrelation of the frame.
    """
    ac = []
    for i in range(N):
        if i == 0:
            ac.append(np.sum(x * x))
        else:
            ac.append(np.sum(x[0:-i] * x[i:]))
    return np.array(ac)


def detect_peak(ac):
    """
    Detect peaks of the autocorralation of the frame.

    Args:
        ac (ndarray): Autocorrelation of the frame.

    Returns:
        ndarray: List of the index which have peak and its value.
    """
    peak = np.zeros(len(ac) - 1)
    for i in range(len(ac) - 2):
        if ac[i] < ac[i + 1] and ac[i + 1] > ac[i + 2]:
            peak[i + 1] = ac[i + 1]
    if len(peak) != 0:
        m0 = np.argmax(peak)
    else:
        m0 = 0
    return m0


def f_0autocor(x, samplerate, overlap, framesize):
    """
    Calculate the fundamental frequency series of the input waveform.

    Args:
        x (ndarray): Input wave.
        samplerate (int): Samplerate of input wave.
        overlap (float): Overlap rate.
        framesize (int): Window size.

    Returns:
        ndarray: Estimated f0 series.
    """
    N = len(x)
    step = int(framesize * (1 - overlap))
    # Calculate the number of times to do windowing
    split_time = int(N / step)
    window = np.hamming(framesize)
    f0_series = []
    for t in range(split_time):
        if t * step + framesize > N:
            x = np.append(x, np.zeros(t * step + framesize - N))
        frame = x[t * step : t * step + framesize]
        # Calculate autocorrelation of the frame.
        ac = calc_ac(frame * window, framesize)
        # Detect peaks from autocorrelation list.
        peak = detect_peak(ac)
        # Calculate estimated f0 frequency.
        if peak == 0:
            f0_frequency = 0
        else:
            f0_frequency = samplerate / peak
        f0_series.append(f0_frequency)
    return np.array(f0_series)


def main():
    """Estimate f0 frequency."""
    parser = argparse.ArgumentParser(description="This program estimates f0 frequency.")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=128, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    parser.add_argument("path", help="the path to the audio file")
    args = parser.parse_args()

    sound_file = args.path
    framesize = args.framesize
    overlap = args.overlap
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)

    ac = f_0autocor(data, samplerate, overlap, framesize)

    # Plot re-synthesized waveform
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Create a time axis
    step = int(framesize * (1 - overlap))
    t = np.arange(0, int(len(data) / step) * step, step) / samplerate
    ax.plot(t, ac, label="autocorrelation")
    ax.set_title("Comparison of input wave and autocorrelation")
    ax.set_xlabel("Time[s]")
    ax.set_ylabel("Magnitude")
    ax.legend(loc=0)
    fig.savefig("f0series_auto.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
