#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for linear predictive coding analysis."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable

import autocorrelation as auto


def lpc(x, p, framesize):
    """
    Calculate linear prediction coefficients by using the Levinson-Durbin algorithm.

    Args:
        x (ndarray): Input wave.

    Returns:
        ndarray: Cepstrum of the input wave.
    """
    
    r = auto.calc_ac(x, len(x))[:p+1]
    a, e = lev_dur(r, p)
    w, h = scipy.signal.freqz(np.sqrt(e), a, framesize, "whole")
    spec_env = 20 * np.log10(np.abs(h))
    return spec_env

def lev_dur(r, p):
    """
    Calculate linear prediction coefficients by using the Levinson-Durbin algorithm.

    Args:
        r (_type_): _description_
        p (_type_): _description_

    Returns:
        _type_: _description_
    """
    if p == 1:
        alpha = np.array([1, -r[1]/r[0]])
        e = np.dot(r[:2], alpha)
    else:
        alpha_s, e_s = lev_dur(r, p-1)
        wp = np.sum(alpha_s[:p]*r[p:0:-1], axis=0)
        kp = wp / e_s
        alpha_u = np.append(alpha_s, 0)
        alpha_rev = np.flipud(alpha_u)
        alpha = alpha_u - kp * alpha_rev
        e = e_s - kp * wp
    return alpha, e

def main():
    """Estimate f0 frequency."""
    parser = argparse.ArgumentParser(
        description="This program estimates f0 frequency."
    )
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=512, type=int
    )
    parser.add_argument(
        "-p", help="the degree for lpc", default=51, type=int
    )
    parser.add_argument("path", help="the path to the audio file")
    args = parser.parse_args()

    sound_file = args.path
    framesize = args.framesize
    p = args.p
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)
    # Calculate the playback time of the input waveform
    time = len(data) / samplerate

    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
    freq = np.fft.rfftfreq(framesize)
    window = np.hamming(framesize)
    data_windowed = data[:framesize] * window
    log = 20 * np.log10(np.abs(np.fft.rfft(data_windowed)))
    cep_env = lpc(data_windowed, p, framesize)
    ax2.plot(freq, log, label="Spectrum")
    ax2.plot(freq, cep_env[:len(log)], label="Cepstrum")
    ax2.set_title("Spectrum envelope")
    ax2.set_xlabel("Frequency[Hz]")
    ax2.set_ylabel("Amplitude[dB]")
    ax2.legend(loc=0)
    plt.tight_layout()
    plt.show()
    plt.close()


if "__main__" == __name__:
    main()
