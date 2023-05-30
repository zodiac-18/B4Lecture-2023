"""Calculate and plot envelope."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import F0


def cepstrum_envelope(data, nfft, lifter):
    """Estimate envelope with cepstrum.

    Args:
        data (ndarray): target data
        nfft (int): number of FFT points
        lifter (int): number of liftering

    Returns:
        ndarray: array cepstrum envelope
    """
    ceps = F0.cepstrum(data)
    ceps[lifter: nfft - lifter + 1] = 0
    cep_env = 20 * np.fft.fft(ceps).real

    return cep_env


def Levinson_durbin(r, order):
    """Calculate Levinson durbin.

    Args:
        r (ndarray): target data
        order (int): order

    Returns:
        ndarray: array
    """
    a = np.zeros(order + 1)
    e = np.zeros(order + 1)
    a[0] = 1
    a[1] = -r[1] / r[0]
    e[0] = r[0]
    e[1] = r[0] + r[1] * a[1]

    for i in range(2, order + 1):
        k = -np.dot(a[:i], r[i:0:-1]) / e[i - 1]
        a[: i + 1] += k * a[: i + 1][::-1]
        e[i] = e[i - 1] * (1 - k**2)
    return a, e[-1]


def lpc(data, nfft, dimension):
    """Calcurate LPC.

    Args:
        data (ndarray): target data
        nfft (int): number of FFT points
        dimension (int): dimension

    Returns:
        ndarray: data of LPC
    """
    win = np.hanning(nfft)
    data = data[:nfft] * win
    acr = F0.AutoCorrelation(data)
    acr = acr[: len(acr) // 2]
    a, e = Levinson_durbin(acr, dimension)
    w, h = signal.freqz(np.sqrt(e), a, nfft, True)
    lpc_data = 20 * np.log10(np.abs(h))
    return lpc_data


def plot_envelope(data, nfft, fs, lifter, dimension):
    """Plot envelope of cepstrum and LPC.

    Args:
        data (ndarray): target data
        nfft (int): number of FFT points
        fs (int): sampling frequency
    """
    win = np.hanning(nfft)
    spec = np.fft.rfft(data[:nfft] * win)
    log_data = 20 * np.log10(
            np.abs(spec) + np.finfo(np.float64).eps
        )
    cep_data = cepstrum_envelope(data[:nfft] * win, nfft, lifter)
    lpc_data = lpc(data, nfft, dimension)
    freq = np.fft.fftfreq(nfft, d=1.0 / fs)
    freq = np.arange(nfft // 2 + 1) * (fs / nfft)
    ceps = F0.cepstrum(data[:nfft] * win)
    t = np.arange(nfft) * 1000 / fs

    To_l = int(fs / 800)  # 基本周波数の推定範囲の上限を800Hzとする
    To_h = int(fs / 40)  # 基本周波数の推定範囲の下限を40Hzとする
    ylim = np.max(ceps[To_l:To_h])  # 基本周期のピーク
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, ceps, c="red")
    ax.set_ylim([-0.10, ylim + 0.10])  # 基本周期のピーク+0.2まで表示
    ax.set_xlim([0, 40])  # 30msまで表示
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("log amplitude")

    plt.figure()
    plt.plot(
        freq[: nfft // 2],
        log_data[: nfft // 2],
        label="Spectrum"
    )
    plt.plot(
        freq[: nfft // 2],
        cep_data[: nfft // 2],
        label="Cepstrum Envelope"
    )
    plt.plot(
        freq[: nfft // 2],
        lpc_data[: nfft // 2],
        label="LPC"
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [db]")
    plt.savefig("envelope")
    plt.legend()
    plt.show()
