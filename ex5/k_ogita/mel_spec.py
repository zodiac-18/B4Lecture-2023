#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for cepstrum analysis."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack.realtransforms
import soundfile as sf

import spec as s


class MelFilterBank:
    def __init__(self, fs, f0, degree, framesize, overlap):
        self.fs = fs
        self.f0 = f0
        self.degree = degree
        self.framesize = framesize
        self.overlap = overlap
        self.filterbank = None
        self.f_centers = None

    def calc_mo(self):
        return 1000 * 1 / np.log10(1000 / self.f0 + 1.0)

    def hz2mel(self, f):
        m0 = self.calc_mo()
        return m0 * np.log10(f / self.f0 + 1.0)

    def mel2hz(self, m):
        m0 = self.calc_mo()
        return self.f0 * (np.power(10, m / m0) - 1.0)

    def melfilterbank(self):
        fmax = self.fs // 2
        melmax = self.hz2mel(fmax)
        nummax = self.framesize // 2
        dmel = melmax / (self.degree + 1)
        mel_centers = np.arange(self.degree + 2) * dmel
        f_centers = self.mel2hz(mel_centers)
        bin = np.floor((self.framesize) * f_centers / self.fs)
        filterbank = np.zeros((self.degree, nummax + 1))
        for m in range(1, self.degree + 1):
            bank_start, bank_center, bank_finish = (
                int(bin[m - 1]),
                int(bin[m]),
                int(bin[m + 1]),
            )
            for k in range(bank_start, bank_center):
                filterbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(bank_center, bank_finish):
                filterbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        self.filterbank, self.f_centers = filterbank, f_centers
        return filterbank, f_centers

    def calc_mfcc(self, data, ncepstrum):
        N = len(data)
        emp_data = pre_emphasis(data, p=0.97)
        step = int(self.framesize * (1 - self.overlap))
        # Calculate the number of times to do windowing
        split_time = int(N / step)
        window = np.hamming(self.framesize)
        filterbank, f_centers = self.melfilterbank()
        mfcc = np.zeros((split_time, ncepstrum))
        mel_spectrum = np.zeros((split_time))
        for t in range(split_time):
            if t * step + self.framesize > N:
                emp_data = np.append(emp_data, np.zeros(t * step + self.framesize - N))
            frame = emp_data[t * step : t * step + self.framesize] * window
            fft_frame = np.abs(np.fft.rfft(frame))
            comp_fbank = np.dot(fft_frame, filterbank.T)
            comp_fbank = np.where(comp_fbank == 0, np.finfo(float).eps, comp_fbank)
            mel_spectrum = 20 * np.log10(comp_fbank)
            cepstrum = scipy.fftpack.dct(mel_spectrum, type=2, norm="ortho")
            mfcc[t, :] = cepstrum[1 : ncepstrum + 1]
        return mfcc

    def delta(self, mfcc, l=2):
        mfcc_pad = np.pad(mfcc, [(0, 0), (l, l + 1)], "edge")
        k_square = np.sum(np.power(np.arange(-l, l + 1), 2))
        k_sequence = np.arange(-l, l + 1)
        delta_mfcc = np.zeros_like(mfcc_pad)
        for i in range(mfcc.shape[1]):
            delta_mfcc[:, i] = np.dot(k_sequence, mfcc_pad[:, i : i + l * 2 + 1].T)
        delta_mfcc = delta_mfcc / k_square
        return delta_mfcc


def pre_emphasis(data, p=0.97):
    N = len(data)
    pre_data = np.zeros(N)
    for i in range(1, N):
        pre_data[i] = data[i] - p * data[i - 1]
    return pre_data


def main():
    """Estimate f0 frequency."""
    parser = argparse.ArgumentParser(description="This program estimates f0 frequency.")
    parser.add_argument(
        "-f", "--framesize", help="the size of window", default=512, type=int
    )
    parser.add_argument(
        "-o", "--overlap", help="the rate of overlap", default=0.8, type=float
    )
    parser.add_argument(
        "-t", "--tap", help="the number of taps of lifter", default=51, type=int
    )
    parser.add_argument(
        "-n",
        "--ncepstrum",
        help="upper bound on the order of the lower order cepstrum to be cut out to find the MFCC",
        default=12,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--degree",
        help="the number of degrees to compress the frequency domain",
        default=20,
        type=int,
    )
    parser.add_argument("-fo", help="frequency parameter", default=700, type=int)
    parser.add_argument("path", help="the path to the audio file")
    args = parser.parse_args()

    sound_file = args.path
    framesize = args.framesize
    overlap = args.overlap
    ncepstrum = args.ncepstrum
    degree = args.degree
    f0 = args.fo
    # Get waveform and sampling rate from the audio file
    data, samplerate = sf.read(sound_file)

    MFB = MelFilterBank(samplerate, f0, degree, framesize, overlap)
    mfcc = MFB.calc_mfcc(data, ncepstrum)
    filterbank = MFB.filterbank

    delt = MFB.delta(mfcc)
    deltdelt = MFB.delta(delt)

    cmap_keyword = "jet"
    cmap = plt.get_cmap(cmap_keyword)

    fig0 = plt.figure(figsize=(15, 10))
    ax0 = fig0.add_subplot(111)
    f = np.arange(0, samplerate // 2, (samplerate // 2) / (framesize // 2 + 1))
    for i in range(1, filterbank.shape[0]):
        ax0.plot(f, filterbank[i, :], color=cmap((i + 1) / filterbank.shape[0]))
    ax0.set_title("Mel Filter Bank")
    ax0.set_xlabel("Frequency [Hz]")
    ax0.set_ylabel("Magnitude")
    fig0.savefig("filter_bank.png")

    fig1 = plt.figure(figsize=(15, 10))
    ax1_1 = fig1.add_subplot(2, 1, 1)
    s.draw_spectrogram(
        data,
        ax=ax1_1,
        framesize=framesize,
        y_limit=samplerate // 2,
        time=len(data) / samplerate,
        overlap=overlap,
        samplerate=samplerate,
    )
    ax1_1.set_title("Original Signal")

    ax1_2 = fig1.add_subplot(2, 1, 2)
    s.draw_spectrogram(
        mfcc,
        ax=ax1_2,
        framesize=framesize,
        y_limit=samplerate // 2,
        time=len(data) / samplerate,
        overlap=overlap,
        samplerate=samplerate,
        is_spec=True,
    )
    ax1_2.set_title("MFCC")
    fig1.savefig("mfcc.png")

    fig2 = plt.figure(figsize=(15, 10))
    ax2_1 = fig2.add_subplot(2, 1, 1)
    img = ax2_1.imshow(
        delt[:ncepstrum].T,
        aspect="auto",
        extent=[0, len(data) / samplerate, 0, ncepstrum],
        cmap="rainbow",
        origin="lower",
    )
    ax2_1.set(
        title="$\Delta$MFCC sequence",
        xlabel="time[s]",
        ylabel="$\Delta$MFCC",
        yticks=range(0, 13, 4),
    )
    fig2.colorbar(img, aspect=10, pad=0.01, ax=ax2_1, format="%+2.f dB")
    ax2_1.set_ylabel("$\Delta$MFCC")

    ax2_2 = fig2.add_subplot(2, 1, 2)
    img = ax2_2.imshow(
        deltdelt[:ncepstrum].T,
        aspect="auto",
        extent=[0, len(data) / samplerate, 0, ncepstrum],
        cmap="rainbow",
        origin="lower",
    )
    ax2_2.set(
        title="$\Delta\Delta$MFCC sequence",
        xlabel="time[s]",
        ylabel="$\Delta\Delta$MFCC",
        yticks=range(0, 13, 4),
    )
    fig2.colorbar(img, aspect=10, pad=0.01, ax=ax2_2, format="%+2.f dB")
    ax2_2.set_ylabel("$\Delta\Delta$MFCC")

    fig2.savefig("delta.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
