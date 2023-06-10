#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for cepstrum analysis."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import autocorrelation as auto


def cepstrum(x):
    """
    Calculate a cepstrum of the input wave.

    Args:
        x (ndarray): Input wave.

    Returns:
        ndarray: Cepstrum of the x.
    """
    spec = np.fft.rfft(x)
    log_power_spec = np.log10(spec)
    cep = np.real(np.fft.irfft(log_power_spec))
    return cep


def cep_specenv(x, tap):
    """
    Calculate a spectrum envelope of the input wave using cepstrum analysis.

    Args:
        x (ndarray): Input wave.
        tap (int): The number of taps of cepstrum.

    Returns:
        ndarray: Spectrum envelope of the input wave.
    """
    cep = cepstrum(x)
    # Extract lower-order cepstrum.
    cep[tap:-tap] = 0
    low_cep = cep
    # Calculate spectrum envelope.
    spec_env = 20 * np.real(np.fft.rfft(low_cep))
    return spec_env


def f_0cep(x, tap, overlap, framesize, samplerate):
    """
    Calculate a f0 frequency of the input wave using cepstrum analysis.

    Args:
        x (ndarray): Input wave.
        tap (int): The number of taps of cepstrum.

    Returns:
        ndarray: f0 series of the input wave.
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
        # Calculate cepstrum of windowed frame.
        cep = cepstrum(frame * window)
        # Extract higher-order cepstrum.
        high_cep = cep[tap : len(cep) // 2]
        # Detect peak from higher-order cepstrum.
        peak = auto.detect_peak(high_cep)
        # Calculate estimated f0 frequency.
        if peak == 0:
            f0_frequency = 0
        else:
            f0_frequency = samplerate / (tap + peak)
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
    parser.add_argument(
        "-t", "--tap", help="the number of taps of lifter", default=51, type=int
    )
    parser.add_argument("path", help="the path to the audio file")
    args = parser.parse_args()

    sound_file = args.path
    framesize = args.framesize
    overlap = args.overlap
    tap = args.tap
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)

    ac = f_0cep(data, tap, overlap, framesize, samplerate)

    # Plot re-synthesized waveform
    _, ax1 = plt.subplots(1, 1, figsize=(20, 10))

    # Create a time axis
    step = int(framesize * (1 - overlap))
    t_0 = np.arange(0, len(data)) / samplerate
    t_1 = np.arange(0, int(len(data) / step) * step, step) / samplerate
    ax1.plot(t_0, data, label="original wave")
    ax1.plot(t_1, ac, label="autocorrelation")
    ax1.set_title("Comparison of input wave and autocorrelation")
    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("Magnitude")
    ax1.legend(loc=0)

    framesize = 512
    _, ax2 = plt.subplots(1, 1, figsize=(20, 10))
    freq = np.fft.rfftfreq(framesize)
    window = np.hamming(framesize)
    data_windowed = data[:framesize] * window
    log = 20 * np.log10(np.abs(np.fft.rfft(data_windowed)))
    cep_env = cep_specenv(data_windowed, tap=51)
    print(cep_env)
    ax2.plot(freq, log, label="Spectrum")
    ax2.plot(freq, cep_env[: len(log)], label="Cepstrum")
    ax2.set_title("Spectrum envelope")
    ax2.set_xlabel("Frequency[Hz]")
    ax2.set_ylabel("Amplitude[dB]")
    ax2.legend(loc=0)
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
