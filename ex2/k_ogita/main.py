#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make a digital filter."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable

import spec as s


def sinc(x):
    """
    Sinc function.

    Args:
        x (float): Input x.

    Returns:
        float: Output of sinc function.
    """
    if x == 0.0:
        return 1.0
    else:
        return np.sin(x) / x


def conv(sig1, sig2):
    """
    Convolute two signals.

    Args:
        sig1 (ndarray): Input signal.
        sig2 (ndarray): Input signal.

    Returns:
        ndarray: Convoluted siganl.
    """
    conv_length = len(sig1) + len(sig2) - 1
    conv_sig = np.zeros(conv_length, dtype=np.float32)

    for n in range(len(sig1)):
        conv_sig[n : n + len(sig2)] += sig1[n] * sig2

    return conv_sig


def bef(cf1, cf2, tap, samplerate):
    """
    Create a base elimination filter.

    Args:
        cf1 (float): cut-off frequency (cf1 < cf2)
        cf2 (float): cut-off frequency (cf1 < cf2)
        tap (int): the number of taps on filter
        samplerate (int) : samplerate of the input

    Returns:
        ndarray: Band elimination filter.
    """
    if tap % 2 != 0:
        tap += 1
    window = np.hamming(tap + 1)

    cw1 = 2 * np.pi * cf1 / samplerate
    cw2 = 2 * np.pi * cf2 / samplerate

    h = []
    for i in range(-tap // 2, tap // 2 + 1):
        h.append(sinc(np.pi * i) + (cw1 * sinc(cw1 * i) - cw2 * sinc(cw2 * i)) / np.pi)
    return np.array(h) * window


def main():
    """Run through the input audio through the filter."""
    parser = argparse.ArgumentParser(
        description="This program generates filtered audiowave"
    )
    parser.add_argument("path", help="the path to the audio file")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=2048, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    parser.add_argument(
        "-cf",
        "--cutoff",
        help="cut-off frequency (two values are required)",
        default=[2000, 4000],
        type=int,
        nargs=2,
    )
    parser.add_argument(
        "-t",
        "--tap",
        help="the number of taps on filter",
        default=101,
        type=int,
    )
    args = parser.parse_args()

    sound_file = args.path
    # Window size
    framesize = args.framesize
    # Rate of overlap
    overlap = args.overlap
    # Cut-off frequency
    cf1, cf2 = args.cutoff[0], args.cutoff[1]
    if cf1 >= cf2:
        raise ValueError("'cf2' must be greater than 'cf1'")
    # Taps of filter
    tap = args.tap
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate

    # Generate a spectrogram from the input waveform
    original_spec = s.stft(data, framesize, overlap)
    # Convert the spectrogram into a dB-scaled spectrogram
    db_original_spec = s.spectrogram_to_db(original_spec, framesize)

    # Generate a band elimination filter
    filter = bef(cf1, cf2, tap, samplerate)
    # Convolute the input signal and the filter
    filtered_wave = conv(data, filter)
    # Generate a spectrogram from the filtered wave
    filtered_spec = s.stft(filtered_wave, framesize, overlap)
    # Convert the spectrogram into a dB-scaled spectrogram
    db_filtered_spec = s.spectrogram_to_db(filtered_spec, framesize)

    # Convert　the filter to frequency domain
    filter_freq = np.fft.fft(filter)
    filter_abs = np.abs(filter_freq)[: tap // 2 + 1]
    # filter_amp = 20 * np.log10(filter_abs)
    filter_phase = np.unwrap(np.angle(filter_freq))[: tap // 2 + 1] * 180 / np.pi

    freq = np.fft.fftfreq(tap, d=1.0 / samplerate)[: tap // 2 + 1] / 1000

    # Plot original spectrogram
    fig1, ax1 = plt.subplots(2, 1, figsize=(10, 10))
    # Plot frequency response of BEF
    ax1[0].plot(freq, filter_abs)
    ax1[0].set_title("Frequency response of BEF (Amplitude)")
    ax1[0].set_xlabel("Frequency [kHz]")
    ax1[0].set_ylabel("Amplitude [dB]")

    ax1[1].plot(freq, filter_phase)
    ax1[1].set_title("Frequency responce of BEF (Phase)")
    ax1[1].set_xlabel("Frequency [kHz]")
    ax1[1].set_ylabel("Phase (rad)")

    fig1.savefig("freq_responce.png")

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 10))
    # Plot original spectrogram
    ax2[0].set_title("Original spectrogram")
    # Transpose the spectrogram to make the vertical axis
    # the frequency and the horizontal axis the time
    im2 = ax2[0].imshow(
        db_original_spec.T,
        extent=[0, time, 0, samplerate / 2000],
        aspect="auto",
        origin="lower",
    )
    divider = make_axes_locatable(ax2[0])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax2[0].set_xlabel("Time[s]")
    ax2[0].set_ylabel("Frequency [kHz]")
    ax2[0].set_xlim(0, time)
    ax2[0].set_ylim(0, samplerate / 2000)
    fig2.colorbar(im2, ax=ax2[0], format="%+2.f dB", cax=cax)

    # Plot filtered spectrogram
    ax2[1].set_title("Filtered spectrogram")
    # Transpose the spectrogram to make the vertical axis
    # the frequency and the horizontal axis the time
    im2 = ax2[1].imshow(
        db_filtered_spec.T,
        extent=[0, time, 0, samplerate / 2000],
        aspect="auto",
        origin="lower",
    )
    divider = make_axes_locatable(ax2[1])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax2[1].set_xlabel("Time[s]")
    ax2[1].set_ylabel("Frequency [kHz]")
    ax2[1].set_xlim(0, time)
    ax2[1].set_ylim(0, samplerate / 2000)
    fig2.colorbar(im2, ax=ax2[1], format="%+2.f dB", cax=cax)

    fig2.savefig("spectrogram.png")

    istft_wave = s.istft(filtered_spec, framesize, overlap)

    # Create audio files from re-synthesized waveforms
    sf.write("re-miku.wav", istft_wave, samplerate)

    # Plot re-synthesized waveform
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10))

    # Create a time axis
    t_0 = np.arange(0, len(data)) / samplerate
    t_1 = np.arange(0, len(istft_wave)) / samplerate
    ax3.plot(t_0, data, label="original wave")
    ax3.plot(t_1, istft_wave, label="filtered wave")
    ax3.set_title("Comparison of waveforms before and after filtering")
    ax3.set_xlabel("Time[s]")
    ax3.set_ylabel("Magnitude")
    ax3.legend(loc=0)
    fig3.savefig("wave.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
