#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate spectrogram and re-synthesized waveform."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable


def stft(data, framesize, overlap):
    """
    Compute the short-time Fourier transform (STFT) of the input waveform.

    Args:
        data (ndarray): Input waveform.
        framesize (int): Window size.
        overlap (float): Overlap rate. Takes a value between 0 and 1.

    Returns:
        ndarray: Spectrogram of waveform.
    """
    # Use a hamming window
    window = np.hamming(framesize)
    step = int(framesize * (1 - overlap))
    # Calculate the number of times to do windowing
    split_time = int(data.shape[0] // step) - 1
    # Make an empty list to store the spectrogram
    stft_result = []
    pos = 0
    # Apply FFT to windowed frames
    for _ in range(split_time):
        if pos + framesize > data.shape[0]:
            break
        frame = np.fft.fft(data[int(pos) : int(pos + framesize)] * window)
        stft_result.append(frame)
        pos += step

    return np.array(stft_result)


def istft(spec, framesize, overlap):
    """
    Compute the Inverse short-time Fourier transform (ISTFT) of spectrogram.

    Args:
        spec (ndarray): Input spectrogram.
        framesize (int): Window size.
        overlap (float): Overlap rate. Takes a value between 0 and 1.

    Returns:
        ndarray: Re-synthesized waveform.
    """
    window = np.hamming(framesize)
    step = int(framesize * (1 - overlap))
    # Calculate the number of samples in the re-synthesized waveform
    num_istft = spec.shape[0] * step + framesize
    # Create an array to store the re-synthesized waveforms
    istft_result = np.zeros(int(num_istft))
    pos = 0
    for i in range(spec.shape[0]):
        # Compute the iFFT of the spectrum
        frame = np.fft.ifft(spec[i, :])
        frame = np.real(frame) / window
        # Add unwindowed frames to the array
        istft_result[int(pos) : int(pos + step)] += frame[0:step]
        pos += step

    return istft_result


def spectrogram_to_db(spec, framesize):
    """
    Convert a spectrogram into a dB-scaled spectrogram.

    Args:
        spec (ndarray): Spectrogram.
        framesize (int): Window size

    Returns:
        ndarray: dB-scaled spectrogram.
    """
    return 20 * np.log10(np.abs(spec[:, : framesize // 2 + 1]))


def draw_spectrogram(
    data, ax, framesize, time, y_limit, overlap, samplerate, is_spec=False
):
    """
    Draw spectrogram of the data.

    Args:
        data (ndarray): Input data.
        ax (matplotlib.axes._axes.Axes): Axes.
        framesize (int): Framesize of the window.
        time (int): The length of the input data.(sec)
        y_limit (int): The upper limit of y-axis.
        overlap (float): Overlap rate.
        samplerate (int): Samplerate of the input data.
    """
    if not is_spec:
        spectrogram = stft(data, framesize, overlap)
        spectrogram_amp = 20 * np.log10(
            np.abs(spectrogram[:, : int(framesize // 2 + 1)])
        )
    else:
        spectrogram_amp = data
    im = ax.imshow(
        spectrogram_amp.T,
        extent=[0, time, 0, samplerate // 2],
        aspect="auto",
        origin="lower",
        cmap="rainbow"
        
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax.set_ylim(0, min(samplerate // 2, y_limit))
    #ßax.set_xlabel("Time[s]")
    ax.set_ylabel("Frequency[Hz]")
    ax.set_title("Spectrogram")
    plt.colorbar(im, ax=ax, format="%+2.f dB", cax=cax)
    return None


def main():
    """Create a spectrogram from the waveform."""
    parser = argparse.ArgumentParser(
        description="This program generates soundwave and re-synthesized waveform from input"
    )
    parser.add_argument("path", help="the path to the audio file")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=2048, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    args = parser.parse_args()

    sound_file = args.path
    # Window size
    framesize = args.framesize
    # Rate of overlap
    overlap = args.overlap
    print(overlap, framesize)
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate
    # Compute the spectrogram of the waveform
    spectrogram = stft(data, framesize, overlap)

    # Plot original waveform
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    # Create a time axis
    t_1 = np.arange(0, len(data)) / samplerate
    ax1.plot(t_1, data)
    plt.title("Original signal")
    plt.xlabel("Time[s]")
    plt.ylabel("Magnitude")

    # Plot spectrogram
    # Calculate the logarithm of the spectrogram for plotting
    spectrogram_amp = 20 * np.log10(np.abs(spectrogram[:, : int(framesize // 2 + 1)]))
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Spectrogram")
    # Transpose the spectrogram to make the vertical axis
    # the frequency and the horizontal axis the time
    im = ax2.imshow(
        spectrogram_amp.T,
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
    fig.colorbar(im, ax=ax2, format="%+2.f dB", cax=cax)

    # Compute the original waveform from the spectrogram
    istft_wave = istft(spectrogram, framesize, overlap)
    # Create audio files from re-synthesized waveforms
    sf.write("re-miku.wav", istft_wave, samplerate)

    # Plot re-synthesized waveform
    ax3 = fig.add_subplot(3, 1, 3)
    # Create a time axis
    t_3 = np.arange(0, len(istft_wave)) / samplerate
    # if len(istft_wave) > len(data):
    #    data = np.append(data, [0]*(len(istft_wave)-len(data)))
    # else:
    #    istft_wave = np.append(istft_wave, [0]*(len(data)-len(istft_wave)))
    # print(np.mean((data-istft_wave)**2))
    ax3.plot(t_3, istft_wave)
    plt.title("Re-Synthesized signal")
    plt.xlabel("Time[s]")
    plt.ylabel("Magnitude")
    plt.savefig("wave.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
