#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Estimate f0 frequency."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import autocorrelation as auto
import cepstrum as c
import spec as s


def main():
    """Estimate f0 frequency."""
    parser = argparse.ArgumentParser(description="This program estimates f0 frequency.")
    parser.add_argument("path", help="the path to the audio file")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=512, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    parser.add_argument(
        "-y",
        "--ylim",
        help="upper limit of y-axis for the graph",
        default=3000,
        type=int,
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
    # Rate of overlap
    overlap = args.overlap
    # Taps of lifter
    tap = args.tap
    # Upper limit of y-axis.
    y_limit = args.ylim
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate

    # Estimate f0 by two different methods.
    ac = auto.f_0autocor(data, samplerate, overlap, framesize)
    cep = c.f_0cep(data, tap, overlap, framesize, samplerate)

    # Plot f0 series with spectrogram.
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    s.draw_spectrogram(
        data,
        ax=ax1,
        framesize=framesize,
        time=time,
        y_limit=y_limit,
        overlap=overlap,
        samplerate=samplerate,
    )
    step = int(framesize * (1 - overlap))
    t = np.arange(0, int(len(data) / step) * step, step) / samplerate
    ax1.plot(t, ac, color="red", label="f0 series(autocorrelation)")
    ax1.set_title("Estimated F0 series by autocorrelation method")
    ax1.legend(loc="upper right")

    s.draw_spectrogram(
        data,
        ax=ax2,
        framesize=framesize,
        time=time,
        y_limit=y_limit,
        overlap=overlap,
        samplerate=samplerate,
    )
    ax2.plot(t, cep, color="red", label="f0 series(cepstrum)")
    ax2.set_title("Estimated F0 series by cepstrum method")
    ax2.legend(loc="upper right")
    fig.savefig("f0_estimation.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
