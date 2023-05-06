#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate spectrogram and re-synthesized waveform."""

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


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
    # Calculate the number of times to do windowing
    split_time = int(data.shape[0] // (framesize * (1 - overlap))) - 1
    # Make an empty list to store the spectrogram
    stft_result = []
    pos = 0
    # Apply FFT to windowed frames
    for _ in range(split_time):
        frame = np.fft.fft(data[int(pos) : int(pos + framesize)] * window)
        stft_result.append(frame)
        pos += framesize * (1 - overlap)

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
    # Calculate the number of samples in the re-synthesized waveform
    num_istft = spec.shape[0] * framesize * (1 - overlap) + framesize
    # Create an array to store the re-synthesized waveforms
    istft_result = np.zeros(int(num_istft))
    pos = 0
    for i in range(spec.shape[0]):
        # Compute the iFFT of the spectrum
        frame = np.fft.ifft(spec[i, :])
        frame = np.real(frame) * window
        # Add windowed frames to the array
        istft_result[int(pos) : int(pos + framesize)] += frame
        pos += framesize * (1 - overlap)

    return istft_result


def main():
    """
    Create a spectrogram from the read waveform and
    calculate the original waveform from the spectrogram.
    """
    sound_file = "miku.wav"
    # Window size
    framesize = 1024
    # Rate of overlap
    overlap = 0.5
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
    spectrogram_amp = np.log(np.abs(spectrogram[:, : int(framesize * (1 - overlap))]))
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Spectrogram")
    # Transpose the spectrogram to make the vertical axis
    # the frequency and the horizontal axis the time
    im = ax2.imshow(
        spectrogram_amp.T, extent=[0, time, 0, samplerate / 2000], aspect="auto"
    )
    ax2.set_xlabel("Time[s]")
    ax2.set_ylabel("Frequency [kHz]")
    ax2.set_xlim(0, time)
    ax2.set_ylim(0, samplerate / 2000)
    fig.colorbar(im, ax=ax2, format="%+2.f dB")

    # Compute the original waveform from the spectrogram
    istft_wave = istft(spectrogram, framesize, overlap)
    # Create audio files from re-synthesized waveforms
    sf.write("re-miku.wav", istft_wave, samplerate)

    # Plot re-synthesized waveform
    ax3 = fig.add_subplot(3, 1, 3)
    # Create a time axis
    t_3 = np.arange(0, len(istft_wave)) / samplerate
    ax3.plot(t_3, istft_wave)
    plt.title("Re-Synhesized signal")
    plt.xlabel("Time[s]")
    plt.ylabel("Magnitude")
    plt.savefig("wave.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
