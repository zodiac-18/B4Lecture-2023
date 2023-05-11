#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make a digital filter."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable

import spec as s


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
    conv_sig = np.zeros(conv_length, dtype = np.float32)
    
    # Apply 0-padding on each signal.
    sig1_pad = np.hstack([sig1, np.zeros(len(sig2) - 1)])
    sig2_pad = np.hstack([sig2, np.zeros(len(sig1) - 1)])
    
    for n in range(conv_length):
        for m in range(len(sig2)):
            conv_sig[n] += sig1_pad[n - m] * sig2_pad[m]

    return conv_sig
    
def bef(cf1, cf2, tap):
    """
    Create a base elimination filter.

    Args:
        cf1 (float): cut-off frequency (cf1 < cf2)
        cf2 (float): cut-off frequency (cf1 < cf2)
        tap (int): the number of taps on filter

    Returns:
        ndarray: Base elimination filter.
    """
    h = np.zeros(tap)
    window = np.hamming(tap)
    
    for n in range(tap):
        if n == 0:
            h[n] = 1 - (cf2 - cf1) / np.pi
        else:
            h[n] = 1 / np.pi * (cf1 * np.sinc(cf1 * n) - cf2 * np.sinc(cf2 * n))
    h_norm = h / np.sum(h)
    h_gain = h_norm * (1 / np.max(np.abs(np.fft.fft(h_norm)))) 
    return h_gain * window


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
        "-cf", "--cutoff", help="cut-off frequency (two values are required)", default=[2000, 4000], type=int, nargs=2
    )
    parser.add_argument(
        "-t", "--tap", help="the number of taps on filter", default=101, type=int,
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
    # Taps on filter
    tap = args.tap
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate
    
    # Generate a spectrogram from the input waveform
    original_spec = s.stft(data, framesize, overlap)
    
    # Convert the spectrogram into a dB-scaled spectrogram
    db_original_spec = s.spectrogram_to_db(original_spec, samplerate)
    
    # Generate a base elimination filter
    filter = bef(cf1, cf2, tap)
    
    # Convolute the input signal and the filter
    filtered_wave = conv(data, filter)
    
    filtered_spec = s.stft(filtered_wave, framesize, overlap)
    
    db_filtered_spec = s.spectrogram_to_db(filtered_spec, samplerate)
    
    # Plot original spectrogram
    fig = plt.figure()
     # Plot re-synthesized waveform
    ax0 = fig.add_subplot(4, 1, 1)
    # Create a time axis
    t_0 = np.arange(0, 101)
    ax0.plot(t_0, filter)
    #t_0 = np.arange(0, len(data)) / samplerate
    #ax0.plot(t_0, data)
    plt.title("filter")
    plt.xlabel("n")
    plt.ylabel("Magnitude")
    # Plot filtered spectrogram
    ax1 = fig.add_subplot(4, 1, 2)
    ax1.set_title("Original spectrogram")
    # Transpose the spectrogram to make the vertical axis
    # the frequency and the horizontal axis the time
    im1 = ax1.imshow(
        db_original_spec.T,
        extent=[0, time, 0, samplerate / 2000],
        aspect="auto",
        origin="lower",
    )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("Frequency [kHz]")
    ax1.set_xlim(0, time)
    ax1.set_ylim(0, samplerate / 2000)
    fig.colorbar(im1, ax=ax1, format="%+2.f dB", cax=cax)

    # Plot filtered spectrogram
    ax2 = fig.add_subplot(4, 2, 3)
    ax2.set_title("Filtered spectrogram")
    # Transpose the spectrogram to make the vertical axis
    # the frequency and the horizontal axis the time
    im2 = ax2.imshow(
        db_filtered_spec.T,
        extent=[0, time, 0, samplerate / 2000],
        aspect="auto",
        origin="lower",
    )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax2.set_xlabel("Time[s]")
    ax2.set_ylabel("Frequency [kHz]")
    ax2.set_xlim(0, time)
    ax2.set_ylim(0, samplerate / 2000)
    fig.colorbar(im2, ax=ax2, format="%+2.f dB", cax=cax)
    
    istft_wave = s.istft(filtered_spec, framesize, overlap)

    # Create audio files from re-synthesized waveforms
    sf.write("re-miku.wav", istft_wave, samplerate)

    # Plot re-synthesized waveform
    ax3 = fig.add_subplot(4, 2, 4)
    # Create a time axis
    t_3 = np.arange(0, len(istft_wave)) / samplerate
    ax3.plot(t_3, istft_wave)
    #t_3 = np.arange(0, len(data)) / samplerate
    #ax3.plot(t_3, data)
    plt.title("Re-Synthesized signal")
    plt.xlabel("Time[s]")
    plt.ylabel("Magnitude")
    plt.savefig("wave.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
