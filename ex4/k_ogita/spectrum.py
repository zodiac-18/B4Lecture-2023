#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Estimate spectrum envelope."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import cepstrum as c
import lpc


def main():
    """Estimate spectrum envelope."""
    parser = argparse.ArgumentParser(
        description="This program estimates spectrum envelope."
    )
    parser.add_argument("path", help="the path to the audio file")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=8192, type=int
    )
    parser.add_argument("-p", help="the degree for lpc", default=81, type=int)
    parser.add_argument(
        "-ti",
        "--time",
        help="start time of the interval for which the spectrul envelope is sought(sec.)",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-t",
        "--tap",
        help="the number of taps on lifter",
        default=51,
        type=int,
    )
    args = parser.parse_args()

    # Path of the sound file.
    sound_file = args.path
    # Window size
    framesize = args.framesize
    # Taps of lifter
    tap = args.tap
    # Start time of the interval for which the spectrul envelope is sought.
    s_time = args.time
    # The degree for lpc
    p = args.p

    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate

    # Raise error if s_time is bigger than the length of the input file.
    if s_time > time:
        raise ValueError("--time must be shorter than the length of the input data.")

    # Plot the spectrul envelopes.
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    freq = np.fft.rfftfreq(framesize, d=1 / samplerate)
    # Use hamming window.
    window = np.hamming(framesize)
    start = int(s_time * samplerate)
    # Output audio file of clipped part.
    sf.write("clipped.wav", data[start : start + framesize], samplerate)
    # Apply window to clipped data.
    data_windowed = data[start : start + framesize] * window

    # Estimate spectrul envelope in two methods, respectively.
    cep_env = c.cep_specenv(data_windowed, tap)
    lpc_env = lpc.lpc(data_windowed, p, framesize)

    # Calculate spectrum of clipped data.
    spectrum = 20 * np.log10(np.abs(np.fft.rfft(data_windowed)))
    ax.plot(freq, spectrum, label="Spectrum")
    ax.plot(freq, cep_env[: len(spectrum)], color="orange", label="Cepstrum")
    ax.plot(freq, lpc_env[: len(spectrum)], color="red", label="LPC")
    ax.set_title("Spectrum envelope")
    ax.set_xlabel("Frequency[Hz]")
    ax.set_ylabel("Amplitude[dB]")
    ax.legend(loc=0)
    fig.savefig("spectrum_envelope.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
