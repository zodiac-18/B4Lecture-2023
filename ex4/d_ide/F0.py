"""Fundamental frequency calculation."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft


def spec_cal(nfft, data, hop_length, window_func):
    """Calculate spectrogram.

    Args:
        nfft (int): number of FFT points
        data (ndarray_): Target data
        hop_length (int): number of samples between successive STFT columns
        window_func (ndarray): window function

    Returns:
        ndarray: spectrogram matrix
    """
    spectrogram = np.zeros(
        (1 + nfft // 2, (len(data) - nfft) // hop_length + 1),
        dtype=np.complex128
    )
    for i in range(spectrogram.shape[1]):
        segment = data[i * hop_length: i * hop_length + nfft] * window_func
        spectrum = fft.fft(segment, n=nfft, axis=0)[: 1 + nfft // 2]
        spectrogram[:, i] = spectrum

    return spectrogram


def AutoCorrelation(data):
    """Calculate autocorrelation function.

    Args:
        data (ndarray): target data

    Returns:
        ndarray: Data after applying autocorrelation function
    """
    length = len(data)
    result = np.zeros(length)
    for m in range(length):
        result[m] = np.dot(
            data[0: length - m],
            data[m:]
        )  # Shift and calculate
    return result


def detect_peak(data):
    """Detect peak of data.

    Args:
        data (ndarray): target data

    Returns:
        int: data of ints
    """
    peak = np.zeros(data.shape[0] - 2)
    for i in range(data.shape[0] - 2):
        if (
            data[i] < data[i + 1] and data[i + 1] > data[i + 2]
        ):  # Compare with previous and next values
            peak[i] = data[i + 1]
    max_i = np.argmax(peak)
    return max_i


def calc_AC(data, fs, nfft, hop_length, win):
    """Calculate autocorrelation.

    Args:
        data (ndarray): target data
        fs (int): sampling frequency
        nfft (int): number of FFT points
        hop_length (int): number of samples between successive STFT columns
        win (ndarray): window function

    Returns:
        ndarray: Array of fundamental frequencies
    """
    shift = (len(data) - nfft) // hop_length
    f0_ac = np.zeros(shift)

    for t in range(shift):
        shift_data = data[t * hop_length: t * hop_length + nfft] * win
        r = AutoCorrelation(shift_data)
        max_i = detect_peak(r)
        if max_i == 0:  # Eliminates the first fundamental frequency
            f0_ac[t] = 0
        else:
            f0_ac[t] = fs / max_i

    return f0_ac


def cepstrum(data):
    """Calculate Cepstrum.

    Args:
        data (ndarray): target data

    Returns:
        ndarray: quefrency
    """
    cr_fft = np.abs(np.fft.rfft(data))
    cr_fft_log = np.log10(
        cr_fft + np.finfo(float).eps
    )  # Add small values to avoid zero
    cr_ifft = np.fft.irfft(cr_fft_log).real
    return cr_ifft


def f0_cepstrum(data, fs, hop_length, nfft, lifter):
    """Calculate fandamental frequency with the Cepstrum method.

    Args:
        data (ndarray): target data
        fs (int): sampling frequency
        hop_length (int): number of samples between successive STFT columns
        nfft (int): number of FFT points
        lifter (int): number of liftering

    Returns:
        ndarray: Array of fundamental frequencies
    """
    n_shift = (len(data) - nfft) // hop_length
    f0_ceps = np.empty(n_shift)
    for i in range(n_shift):
        audio = data[i * hop_length: i * hop_length + nfft] * np.hamming(nfft)
        ceps = cepstrum(audio)
        max_i = detect_peak(ceps[lifter:])
        if max_i == 0:
            f0_ceps[i] = 0
        else:
            f0_ceps[i] = fs / (max_i + lifter)

    return f0_ceps


def plot_F0(data, rate, nfft, hop_length, lifter, win):
    """Plot fandamental frequency.

    Args:
        data (ndarray): target data
        rate (int): sampling frequency
        nfft (int): number of FFT points
        hop_length (int): number of samples between successive STFT columns
        lifter (int): number of liftering
        window_func (ndarray): window function
    """
    n_steps = (len(data) - nfft) // hop_length
    total_frame = hop_length * n_steps + nfft  # number of total frame
    t = total_frame / rate  # total time [s]

    # calculate F0
    f0 = calc_AC(data, rate, nfft, hop_length, win)
    f0_ceps = f0_cepstrum(data, rate, hop_length, nfft, lifter)
    x1_lim = np.linspace(0, t, len(f0))
    x2_lim = np.linspace(0, t, len(f0_ceps))

    # calculate spectrogram
    spectrogram = spec_cal(nfft, data, hop_length, win)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_title("f0(auto-correlation)")
    im = ax1.imshow(
        20 * np.log10(np.abs(spectrogram + np.finfo(float).eps)),
        origin="lower",
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, t, 0, rate // 2],
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax1, format='%+2.0f dB')
    ax1.plot(x1_lim, f0, label="f0(auto-correlation)", color="black")
    ax1.legend()
    plt.savefig("F0(auto-correlation)")
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.set_title("f0(cepstrum)")
    im = ax2.imshow(
        20 * np.log10(np.abs(spectrogram + np.finfo(float).eps)),
        origin="lower",
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, t, 0, rate // 2],
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax2, format='%+2.0f dB')
    ax2.plot(x2_lim, f0_ceps, label="f0(cepstrum)", color="black")
    ax2.legend()
    plt.savefig("F0(cepstrum)")
    plt.tight_layout()
    plt.show()
