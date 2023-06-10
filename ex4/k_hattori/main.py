"""Estimates the fundamental frequendy of sound and spectral envelopes."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import read

import myfunc as mf


def detect_peak(data):
    """
    Find the maximum of the peak values excludeing the first part of the data.

    Args:
        data (ndarray): Target array to find the peak value

    Returns:
        int: Number of elements taking the maximum value
    """
    max_val = 0
    for i in range(len(data) - 10):
        before = data[i + 8]
        handle = data[i + 9]
        after = data[i + 10]
        if before < handle and handle > after and handle > max_val:
            max_val = handle
            max_sample = i + 9

    return max_sample


def AutoCorrelation(data):
    """
    Calculate autocorrelation.

    Args:
        data (ndarray): Array to calculate autocorrelation

    Returns:
        ndarray: Result of autocorrelation
    """
    Correlation = np.zeros(len(data))

    for i in range(len(data)):
        if i == 0:
            Correlation[i] = (np.sum(data * data))
        else:
            Correlation[i] = np.sum(data[0:-i] * data[i:])

    return Correlation


def Correlation_f0predict(data, interval_time, samplerate):
    """
    Estimate fundamental frequency by autocorrelation method.

    Args:
        data (ndarray): Signal to be estimated
        interval_time (float): Interval time to split signal
        samplerate (float): Sampling rate of signal

    Returns:
        ndarray: Fundamental frequency estimated for each divided parcel
    """
    interval = int(samplerate * interval_time)
    N = len(data) // interval
    pos = 0
    max_sample = np.zeros(N)

    for i in range(N):
        window = data[pos: pos + interval]
        correlation = AutoCorrelation(window)
        max_sample[i] = detect_peak(correlation)
        pos += interval

    max_time = max_sample / samplerate
    fund_freq = max_time ** (-1)

    return fund_freq


def cepstrum_f0predict(data, interval_time, samplerate):
    """
    Estimate fundamental frequency by cepstrum method.

    Args:
        data (ndarray): Signal to be estimated
        interval_time (float): Interval time to split signal
        samplerate (float): Sampling rate of signal

    Returns:
        ndarray: Fundamental frequency estimated for each divided parcel
    """
    cutoff = 150
    interval = int(samplerate * interval_time)
    N = len(data) // interval
    pos = 0
    hanning = np.hanning(interval)
    max_sample = np.zeros(N)
    for i in range(N):
        window = data[pos: pos + interval]
        windowed = window * hanning
        fft_window = np.fft.fft(windowed)
        power_spec = 20 * np.log(np.abs(fft_window) ** 2)
        ceps = np.fft.ifft(power_spec)
        ceps[:cutoff] = 0
        ceps[len(ceps) // 2:] = 0
        max_sample[i] = detect_peak(ceps)
        pos += interval

    max_time = max_sample / samplerate
    fund_freq = max_time ** (-1)

    return fund_freq


def envelope_cepstrum(data, interval_time, samplerate):
    """
    Find spectral envelopes using cepstrum method.

    Args:
        data (ndarray): Signal for which the spectral envelope is sought
        interval_time (float): Interval time to split signal
        samplerate (float): Sampling rate of signal

    Returns:
        ndarray: Spectrum of splited signal
        ndarray: Spectral envelopes
    """
    cutoff = 150
    interval = int(samplerate * interval_time)
    N = len(data) // interval
    pos = 0
    hanning = np.hanning(interval)
    spectrum = np.zeros([N, interval])
    envelope = np.zeros([N, interval])
    for i in range(N):
        window = data[pos: pos + interval]
        windowed = window * hanning
        fft_window = np.fft.fft(windowed)

        power_spec = 20 * np.log10(np.abs(fft_window))
        ceps = np.fft.ifft(power_spec)
        ceps[cutoff:] = 0
        split_envelope = np.fft.fft(ceps)
        spectrum[i] = power_spec
        envelope[i] = np.real(split_envelope)
        pos += interval

    return spectrum, envelope


def Levinson_Durbin(data, order, r=None):
    """
    Find coefficients alpha of autoregressive model by Levinson-Durbin method.

    Args:
        data (ndarray): Segmented signal used for this algorithm.
        order (int): order of LPC
        r (ndarray, optional): Teplitz-type matrix used in recursion.

    Returns:
        ndarray: Array of coefficients alpha of the autoregressive model.
    """
    p = order
    if r is None:
        autocorr = np.correlate(data, data, mode="full")
        return Levinson_Durbin(data, order, autocorr[len(data)-1:len(data)+p])

    if p == 1:
        a = np.array([1, -r[1] / r[0]])
        E = a.dot(r[:2])
    else:
        a_next, E_next = Levinson_Durbin(data, p - 1, r)
        k_next = -1 * (a_next.dot(r[p:0:-1])) / E_next
        U = np.append(a_next, 0)
        V = U[::-1]
        a = U + k_next * V
        E = E_next * (1 - k_next * k_next)

    return a, E


def envelope_LPC(data, interval_time, samplerate, order):
    """Find spectral envelopes using cepstrum method.

    Args:
        data (ndarray): Signal for which the spectral envelope is sought
        interval_time (float): Interval time to split signal
        samplerate (float): Sampling rate of signal
        order (int): order of LPC

    Returns:
        ndarray: Frequency of envelope array
        ndarray: Spectral envelopes
    """
    interval = int(samplerate * interval_time)
    N = len(data) // interval
    hanning = np.hanning(interval)
    pos = 0
    envelope = np.zeros([N, interval])
    for i in range(N):
        window = data[pos: pos + interval]
        windowed = window * hanning
        a, E = Levinson_Durbin(windowed, order)
        w, h = signal.freqz(np.sqrt(E), a, interval)
        f = samplerate * w / 2.0 / np.pi
        gain = 20 * np.log10(np.abs(h))
        envelope[i] = gain
        pos += interval

    return f, envelope


def main():
    """
    Find the fundamental frequency and spectral envelope.

    Returns:
        None
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    samplerate, data = read("ex4/k_hattori/ONSEI.wav")
    TOTAL_TIME = len(data) / samplerate

    interval_time = 40 * 10 ** (-3)

    f0_autocorr = Correlation_f0predict(data, interval_time, samplerate)
    f0_cepstrum = cepstrum_f0predict(data, interval_time, samplerate)

    c_spectrum, c_envelope = envelope_cepstrum(data, interval_time, samplerate)
    l_freq, l_envelope = envelope_LPC(data, interval_time, samplerate, 40)

    spectrogram = mf.STFT(data, 1024)
    time = np.linspace(0, TOTAL_TIME, len(f0_autocorr))
    freq = np.linspace(0, samplerate, c_spectrum.shape[1])

    plt.plot(time, f0_autocorr, color="black", label="f0(autocorrelation)")
    plt.legend()
    mf.spectrogram(TOTAL_TIME, samplerate, spectrogram)

    plt.plot(time, f0_cepstrum, color="black", label="f0(cepstrum)")
    plt.legend()
    mf.spectrogram(TOTAL_TIME, samplerate, spectrogram)

    N = 23
    plt.plot(freq[freq <= 8000], c_spectrum[N][freq <= 8000],
             color="gray", label="Spectrum")
    plt.plot(freq[freq <= 8000], c_envelope[N][freq <= 8000],
             color="blue", label="Cepstrum")
    plt.plot(l_freq[l_freq <= 8000], l_envelope[N][l_freq <= 8000],
             color="red", label="LPC")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("log amplitude spectrum [dB]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    exit(1)
