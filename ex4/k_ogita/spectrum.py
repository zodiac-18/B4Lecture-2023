#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Estimate f0 frequency."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cepstrum as c
import lpc as l

def main():
    """Estimate f0 frequency."""
    parser = argparse.ArgumentParser(
        description="This program estimates f0 frequency."
    )
    parser.add_argument("path", help="the path to the audio file")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=512, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    parser.add_argument(
        "-p", help="the degree for lpc", default=101, type=int
    )
    parser.add_argument(
        "-t",
        "--tap",
        help="the number of taps on lifter",
        default=51,
        type=int,
    )
    args = parser.parse_args()

    sound_file = args.path
    # Window size
    framesize = args.framesize
    # Taps of lifter
    tap = args.tap
    
    p = args.p
    
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    freq = np.fft.rfftfreq(framesize, d=1/samplerate)
    window = np.hamming(framesize)
    data_windowed = data[20000:20000+framesize] * window
    spectrum = 20 * np.log10(np.abs(np.fft.rfft(data_windowed)))
    cep_env = c.cep_specenv(data_windowed, tap)
    lpc_env = l.lpc(data_windowed, p, framesize)
    ax.plot(freq, spectrum, label="Spectrum")
    ax.plot(freq, cep_env[:len(spectrum)], color="orange", label="Cepstrum")
    ax.plot(freq, lpc_env[:len(spectrum)], color="red", label="LPC")
    ax.set_title("Spectrum envelope")
    ax.set_xlabel("Frequency[Hz]")
    ax.set_ylabel("Amplitude[dB]")
    ax.legend(loc=0)
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
